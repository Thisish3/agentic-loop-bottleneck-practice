// Step-level simulation of agent sessions sharing one accelerator's KV cache.
//
// Each decode step every running LLM call emits one token; the step lasts itl(batch) plus the
// prefill time of any prompt tokens admitted in that step (chunked prefill folded into the step).
// Between calls a session runs a tool; what happens to its KV meanwhile is the Policy.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_pool.hpp"

namespace kvsim {

struct Call {
  int prompt_tokens;
  int completion_tokens;
  double gap_after_s;     // measured gap to the next call (-1 after the last call)
  std::string tool_class; // "calc" | "lookup" | "web" | "" after the last call
};

using Trace = std::vector<Call>;  // one agent session (one question)

struct ToolLatency {
  bool measured = false;  // true: replay the measured gaps, ignoring the medians below
  double calc_s = 0.01, lookup_s = 0.3, web_s = 3.0;
  double sigma = 0.6;     // lognormal spread of the actual latency around the median
  double median_for(const std::string& cls) const {
    return cls == "web" ? web_s : cls == "lookup" ? lookup_s : calc_s;
  }
};

// Which ready call gets the next slot.
enum class Admission {
  Fifo,     // request-level FIFO (vLLM default): every call is equal, new sessions may evict
  Session,  // continuing sessions first; a new session starts only if it fits without evicting
};

struct Config {
  Policy policy = Policy::Lru;
  Admission admission = Admission::Fifo;
  int kv_tokens = 24064;          // KV capacity (vLLM-reported, 7B-AWQ fp16 KV on the 3060 Ti)
  int block_tokens = 16;
  int max_running = 16;           // max concurrent LLM calls (vLLM --max-num-seqs)
  double itl_base_ms = 14.9;      // measured ITL at batch 1
  double itl_per_seq_ms = 0.44;   // added ITL per extra running sequence (fit to c1 vs c4)
  double prefill_tok_per_s = 1400; // uncached prefill rate (measured: ~2.9k tokens in 2.15 s)
  double arrival_per_s = 1.0;     // Poisson session arrivals
  int sessions = 2000;
  int ctx_pad = 0;  // extra tokens present in every call's prompt (tool schemas, retrieved docs, history)
  ToolLatency tools;
  uint64_t seed = 1;
};

struct Result {
  int sessions_done = 0;
  double makespan_s = 0;
  double latency_p50_s = 0, latency_p95_s = 0, latency_mean_s = 0;
  int64_t prompt_tokens = 0;      // all prompt tokens the calls asked for
  int64_t prefilled_tokens = 0;   // tokens actually computed (prompt minus retained KV)
  int64_t evictions = 0;
  double mean_running = 0;        // time-averaged running calls
  double mean_kv_util = 0;        // time-averaged fraction of KV blocks in use
  double admit_wait_s = 0;        // mean time a ready call waited for memory or a slot
};

Result simulate(const std::vector<Trace>& traces, const Config& cfg);

std::vector<Trace> load_traces(const std::string& tsv_path);

}  // namespace kvsim
