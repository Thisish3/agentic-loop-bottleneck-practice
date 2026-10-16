#include "sim.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <map>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace kvsim {

namespace {

enum class Phase { Pending, Ready, Running, Tool, Done };

struct Session {
  const Trace* trace = nullptr;
  int call = 0;
  Phase phase = Phase::Pending;
  double arrival = 0;
  double ready_since = 0;   // when the current call became ready to run
  double ready_at = 0;      // when the running tool returns
  double idle_since = 0;
  double predicted_gap = 0;
  std::vector<int> blocks;
  int64_t resident_tokens = 0;  // tokens of context whose KV is still in the pool
  int gen_left = 0;
};

std::string classify_tool(const std::string& name) {
  if (name == "-" || name.empty()) return "";
  if (name.find("web_search") != std::string::npos) return "web";
  if (name.find("get_") != std::string::npos) return "lookup";
  return "calc";
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  double idx = q * (v.size() - 1);
  size_t lo = static_cast<size_t>(idx);
  size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

// Median measured gap per tool class: the "hint" an agent framework could attach in measured mode.
std::map<std::string, double> measured_medians(const std::vector<Trace>& traces) {
  std::map<std::string, std::vector<double>> by;
  for (const auto& t : traces)
    for (const auto& c : t)
      if (c.gap_after_s >= 0) by[c.tool_class].push_back(c.gap_after_s);
  std::map<std::string, double> out;
  for (auto& [k, v] : by) out[k] = percentile(v, 0.5);
  return out;
}

}  // namespace

std::vector<Trace> load_traces(const std::string& tsv_path) {
  std::ifstream in(tsv_path);
  if (!in) throw std::runtime_error("cannot open " + tsv_path);
  std::string line;
  std::getline(in, line);  // header
  std::map<int, Trace> by_session;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    int session, call;
    Call c;
    std::string tool;
    if (!(ss >> session >> call >> c.prompt_tokens >> c.completion_tokens >> c.gap_after_s >> tool))
      throw std::runtime_error("bad trace line: " + line);
    c.tool_class = classify_tool(tool);
    by_session[session].push_back(c);
  }
  std::vector<Trace> out;
  for (auto& [id, t] : by_session) out.push_back(std::move(t));
  if (out.empty()) throw std::runtime_error("no sessions in " + tsv_path);
  return out;
}

Result simulate(const std::vector<Trace>& raw_traces, const Config& cfg) {
  std::vector<Trace> traces = raw_traces;
  for (auto& t : traces)
    for (auto& c : t) c.prompt_tokens += cfg.ctx_pad;
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::exponential_distribution<double> interarrival(cfg.arrival_per_s);
  std::uniform_int_distribution<size_t> pick(0, traces.size() - 1);
  const auto measured_hint = measured_medians(traces);

  BlockPool pool(cfg.kv_tokens / cfg.block_tokens, cfg.block_tokens);
  std::vector<Session> ss(cfg.sessions);
  double t = 0;
  for (auto& s : ss) {
    t += interarrival(rng);
    s.arrival = t;
    s.trace = &traces[pick(rng)];
  }

  using Ev = std::pair<double, int>;  // (time, session)
  std::priority_queue<Ev, std::vector<Ev>, std::greater<>> tool_returns;
  std::deque<int> ready;
  std::vector<int> running;
  std::unordered_set<int> holding_idle;  // not running, still holding KV: eviction candidates
  size_t next_arrival = 0;
  int done = 0;
  double now = 0, prefill_tokens_this_step = 0;
  std::vector<double> latencies;
  Result r;
  double busy_time = 0, running_area = 0, kv_area = 0, wait_sum = 0;
  int64_t admitted = 0;

  auto evict_one = [&](int except) {
    std::vector<IdleCandidate> cand;
    std::vector<int> ids;
    for (int id : holding_idle) {
      if (id == except) continue;
      const auto& s = ss[id];
      cand.push_back({id, s.idle_since, s.phase == Phase::Tool ? s.ready_at : now, s.predicted_gap});
      ids.push_back(id);
    }
    int v = pick_victim(cfg.policy, cand, now);
    if (v < 0) return false;
    auto& victim = ss[ids[v]];
    pool.release(victim.blocks);
    victim.resident_tokens = 0;
    holding_idle.erase(ids[v]);
    ++r.evictions;
    return true;
  };

  while (done < cfg.sessions) {
    while (next_arrival < ss.size() && ss[next_arrival].arrival <= now) {
      auto& s = ss[next_arrival];
      s.phase = Phase::Ready;
      s.ready_since = s.arrival;
      ready.push_back(static_cast<int>(next_arrival++));
    }
    while (!tool_returns.empty() && tool_returns.top().first <= now) {
      int id = tool_returns.top().second;
      tool_returns.pop();
      ss[id].phase = Phase::Ready;
      ss[id].ready_since = ss[id].ready_at;
      ready.push_back(id);
    }

    // Admission with head-of-line blocking. Fifo: strict arrival order (vLLM's default).
    // Session: the oldest continuing session goes first; new sessions wait unless memory is free.
    while (static_cast<int>(running.size()) < cfg.max_running && !ready.empty()) {
      auto pos = ready.begin();
      if (cfg.admission == Admission::Session) {
        auto cont = std::find_if(ready.begin(), ready.end(), [&](int i) { return ss[i].call > 0; });
        if (cont != ready.end()) pos = cont;
      }
      int id = *pos;
      auto& s = ss[id];
      const Call& c = (*s.trace)[s.call];
      int64_t ctx = static_cast<int64_t>(c.prompt_tokens) + c.completion_tokens;
      bool may_evict = cfg.admission == Admission::Fifo || s.call > 0 || running.empty();
      bool ok = pool.grow_to(s.blocks, ctx);
      while (!ok && may_evict && evict_one(id)) ok = pool.grow_to(s.blocks, ctx);
      if (!ok) {
        if (running.empty() && pool.used_blocks() == static_cast<int>(s.blocks.size()))
          throw std::runtime_error("a single call does not fit in the KV pool");
        break;
      }
      int64_t cached = std::min<int64_t>(s.resident_tokens, c.prompt_tokens);
      r.prompt_tokens += c.prompt_tokens;
      r.prefilled_tokens += c.prompt_tokens - cached;
      prefill_tokens_this_step += c.prompt_tokens - cached;
      s.gen_left = c.completion_tokens;
      s.phase = Phase::Running;
      holding_idle.erase(id);
      wait_sum += now - s.ready_since;
      ++admitted;
      ready.erase(pos);
      running.push_back(id);
    }

    if (running.empty()) {
      double next = 1e300;
      if (next_arrival < ss.size()) next = ss[next_arrival].arrival;
      if (!tool_returns.empty()) next = std::min(next, tool_returns.top().first);
      if (next == 1e300) throw std::runtime_error("stalled: ready calls but nothing running or pending");
      now = std::max(now, next);
      continue;
    }

    int b = static_cast<int>(running.size());
    double dt = (cfg.itl_base_ms + cfg.itl_per_seq_ms * (b - 1)) / 1000.0 +
                prefill_tokens_this_step / cfg.prefill_tok_per_s;
    prefill_tokens_this_step = 0;
    busy_time += dt;
    running_area += b * dt;
    kv_area += static_cast<double>(pool.used_blocks()) / pool.total_blocks() * dt;
    now += dt;

    std::vector<int> still;
    still.reserve(running.size());
    for (int id : running) {
      auto& s = ss[id];
      if (--s.gen_left > 0) {
        still.push_back(id);
        continue;
      }
      const Call& c = (*s.trace)[s.call];
      s.resident_tokens = static_cast<int64_t>(c.prompt_tokens) + c.completion_tokens;
      if (s.call + 1 == static_cast<int>(s.trace->size())) {
        pool.release(s.blocks);
        s.resident_tokens = 0;
        s.phase = Phase::Done;
        latencies.push_back(now - s.arrival);
        ++done;
        continue;
      }
      double gap, hint;
      if (cfg.tools.measured) {
        gap = c.gap_after_s;
        hint = measured_hint.at(c.tool_class);
      } else {
        hint = cfg.tools.median_for(c.tool_class);
        gap = std::exp(std::log(hint) + cfg.tools.sigma * gauss(rng));
      }
      s.phase = Phase::Tool;
      s.idle_since = now;
      s.ready_at = now + gap;
      s.predicted_gap = hint;
      s.call++;
      tool_returns.push({s.ready_at, id});
      if (cfg.policy == Policy::Drop) {
        pool.release(s.blocks);
        s.resident_tokens = 0;
      } else {
        holding_idle.insert(id);
      }
    }
    running.swap(still);
  }

  r.sessions_done = done;
  r.makespan_s = now;
  r.latency_p50_s = percentile(latencies, 0.5);
  r.latency_p95_s = percentile(latencies, 0.95);
  double sum = 0;
  for (double l : latencies) sum += l;
  r.latency_mean_s = latencies.empty() ? 0 : sum / latencies.size();
  r.mean_running = busy_time > 0 ? running_area / busy_time : 0;
  r.mean_kv_util = busy_time > 0 ? kv_area / busy_time : 0;
  r.admit_wait_s = admitted ? wait_sum / admitted : 0;
  return r;
}

}  // namespace kvsim
