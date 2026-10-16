// Block-granular KV cache pool and the eviction policies for idle agent sessions.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvsim {

// Fixed-size block allocator (vLLM-style paged KV): a free list of block ids.
class BlockPool {
 public:
  BlockPool(int num_blocks, int block_tokens) : block_tokens_(block_tokens) {
    if (num_blocks <= 0 || block_tokens <= 0) throw std::invalid_argument("pool size must be positive");
    free_.reserve(num_blocks);
    for (int i = num_blocks - 1; i >= 0; --i) free_.push_back(i);
    total_ = num_blocks;
  }

  int blocks_for(int64_t tokens) const { return static_cast<int>((tokens + block_tokens_ - 1) / block_tokens_); }
  int free_blocks() const { return static_cast<int>(free_.size()); }
  int used_blocks() const { return total_ - free_blocks(); }
  int total_blocks() const { return total_; }
  int block_tokens() const { return block_tokens_; }

  // Grows `owned` to hold `tokens`; returns false (and changes nothing) when the pool is short.
  bool grow_to(std::vector<int>& owned, int64_t tokens) {
    int need = blocks_for(tokens) - static_cast<int>(owned.size());
    if (need <= 0) return true;
    if (need > free_blocks()) return false;
    for (int i = 0; i < need; ++i) {
      owned.push_back(free_.back());
      free_.pop_back();
    }
    return true;
  }

  void release(std::vector<int>& owned) {
    free_.insert(free_.end(), owned.begin(), owned.end());
    owned.clear();
  }

 private:
  int block_tokens_;
  int total_;
  std::vector<int> free_;
};

// What happens to a session's KV while it waits on a tool call.
enum class Policy {
  Drop,    // free at the end of every LLM call; the next call re-prefills its whole prompt
  Lru,     // keep it; under pressure evict the session idle the longest
  Hint,    // keep it; evict the session whose next call is predicted furthest away (per-tool median latency)
  Oracle,  // keep it; evict by the true time to the next call (upper bound for any predictor)
};

inline Policy parse_policy(const std::string& s) {
  if (s == "drop") return Policy::Drop;
  if (s == "lru") return Policy::Lru;
  if (s == "hint") return Policy::Hint;
  if (s == "oracle") return Policy::Oracle;
  throw std::invalid_argument("unknown policy: " + s);
}

inline const char* policy_name(Policy p) {
  switch (p) {
    case Policy::Drop: return "drop";
    case Policy::Lru: return "lru";
    case Policy::Hint: return "hint";
    case Policy::Oracle: return "oracle";
  }
  return "?";
}

// An idle session holding KV, as seen by the eviction policy.
struct IdleCandidate {
  int session;
  double idle_since;      // when its last LLM call finished
  double true_ready_at;   // when its tool call actually returns
  double predicted_gap;   // the agent framework's estimate for the tool it is running (Hint only)
};

// Picks the victim among idle sessions; returns an index into `c`, or -1 if empty.
inline int pick_victim(Policy p, const std::vector<IdleCandidate>& c, double now) {
  int best = -1;
  double best_key = 0;
  for (int i = 0; i < static_cast<int>(c.size()); ++i) {
    double key = 0;  // larger key = better victim
    switch (p) {
      case Policy::Drop:
      case Policy::Lru: key = now - c[i].idle_since; break;
      case Policy::Hint: key = c[i].idle_since + c[i].predicted_gap - now; break;
      case Policy::Oracle: key = c[i].true_ready_at - now; break;
    }
    if (best < 0 || key > best_key) {
      best = i;
      best_key = key;
    }
  }
  return best;
}

}  // namespace kvsim
