// kv_sim: sweep KV retention policies for agent sessions over replayed NeMo Agent Toolkit traces.
//
//   ./kv_sim --traces traces.tsv --policy drop,lru,hint,oracle --kv-tokens 24064,47264
//            --arrival 0.5,1,2 --tools realistic --web 3 > sweep.csv
//
// Every (policy, kv-tokens, arrival) point is an independent simulation; points run on a thread pool.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "sim.hpp"

using namespace kvsim;

namespace {

template <typename T>
std::vector<T> parse_list(const std::string& s, T (*conv)(const std::string&)) {
  std::vector<T> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(conv(item));
  return out;
}

double to_d(const std::string& s) { return std::stod(s); }
int to_i(const std::string& s) { return std::stoi(s); }
Admission to_admission(const std::string& s) {
  if (s == "fifo") return Admission::Fifo;
  if (s == "session") return Admission::Session;
  throw std::invalid_argument("unknown admission: " + s);
}

void usage() {
  std::fprintf(stderr,
               "usage: kv_sim --traces FILE [--policy drop,lru,hint,oracle] [--kv-tokens N,...]\n"
               "              [--arrival R,...] [--sessions N] [--tools measured|realistic]\n"
               "              [--calc S] [--lookup S] [--web S] [--sigma X] [--max-running N]\n"
               "              [--prefill-rate TOK_PER_S] [--ctx-pad TOKENS] [--admission fifo,session]\n"
               "              [--seeds N] [--threads N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string traces_path;
  std::vector<Policy> policies = {Policy::Drop, Policy::Lru, Policy::Hint, Policy::Oracle};
  std::vector<int> kv_tokens = {24064};
  std::vector<double> arrivals = {1.0};
  std::vector<Admission> admissions = {Admission::Fifo};
  Config base;
  int seeds = 3;
  int threads = static_cast<int>(std::thread::hardware_concurrency());

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--traces") traces_path = val();
    else if (a == "--policy") policies = parse_list<Policy>(val(), parse_policy);
    else if (a == "--kv-tokens") kv_tokens = parse_list<int>(val(), to_i);
    else if (a == "--arrival") arrivals = parse_list<double>(val(), to_d);
    else if (a == "--sessions") base.sessions = std::stoi(val());
    else if (a == "--tools") base.tools.measured = (val() == "measured");
    else if (a == "--calc") base.tools.calc_s = std::stod(val());
    else if (a == "--lookup") base.tools.lookup_s = std::stod(val());
    else if (a == "--web") base.tools.web_s = std::stod(val());
    else if (a == "--sigma") base.tools.sigma = std::stod(val());
    else if (a == "--max-running") base.max_running = std::stoi(val());
    else if (a == "--prefill-rate") base.prefill_tok_per_s = std::stod(val());
    else if (a == "--ctx-pad") base.ctx_pad = std::stoi(val());
    else if (a == "--admission") admissions = parse_list<Admission>(val(), to_admission);
    else if (a == "--seeds") seeds = std::stoi(val());
    else if (a == "--threads") threads = std::stoi(val());
    else { usage(); return 2; }
  }
  if (traces_path.empty()) { usage(); return 2; }

  std::vector<Trace> traces;
  try {
    traces = load_traces(traces_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  struct Point { Config cfg; Result res; std::string err; };
  std::vector<Point> points;
  for (int kv : kv_tokens)
    for (double ar : arrivals)
      for (Admission adm : admissions)
        for (Policy p : policies)
          for (int s = 0; s < seeds; ++s) {
            Point pt;
            pt.cfg = base;
            pt.cfg.kv_tokens = kv;
            pt.cfg.arrival_per_s = ar;
            pt.cfg.admission = adm;
            pt.cfg.policy = p;
            pt.cfg.seed = 1000 + s;  // same seeds across policies: paired comparison
            points.push_back(pt);
          }

  // Work-stealing by atomic index: each point writes only its own slot, so no locks are needed.
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    for (size_t i; (i = next.fetch_add(1)) < points.size();) {
      try {
        points[i].res = simulate(traces, points[i].cfg);
      } catch (const std::exception& e) {
        points[i].err = e.what();
      }
    }
  };
  std::vector<std::thread> pool;
  for (int i = 0; i < std::max(1, threads); ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();

  std::printf("admission,policy,kv_tokens,arrival_per_s,seed,sessions,p50_s,p95_s,mean_s,makespan_s,"
              "prompt_tokens,prefilled_tokens,recompute_share,evictions,mean_running,kv_util,"
              "admit_wait_s,error\n");
  for (const auto& p : points) {
    const auto& r = p.res;
    double share = r.prompt_tokens ? static_cast<double>(r.prefilled_tokens) / r.prompt_tokens : 0;
    std::printf("%s,%s,%d,%.3f,%llu,%d,%.3f,%.3f,%.3f,%.1f,%lld,%lld,%.4f,%lld,%.2f,%.3f,%.3f,%s\n",
                p.cfg.admission == Admission::Fifo ? "fifo" : "session",
                policy_name(p.cfg.policy), p.cfg.kv_tokens, p.cfg.arrival_per_s,
                static_cast<unsigned long long>(p.cfg.seed), r.sessions_done, r.latency_p50_s,
                r.latency_p95_s, r.latency_mean_s, r.makespan_s, static_cast<long long>(r.prompt_tokens),
                static_cast<long long>(r.prefilled_tokens), share, static_cast<long long>(r.evictions),
                r.mean_running, r.mean_kv_util, r.admit_wait_s, p.err.c_str());
  }
  return 0;
}
