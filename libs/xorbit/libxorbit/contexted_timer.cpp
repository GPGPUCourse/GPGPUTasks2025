#include "contexted_timer.h"

#include <iostream>
#include <cmath>

ContextedTimer::ScopedContext::ScopedContext(ContextedTimer& parent, const std::string& name)
    : parent_(parent), name_(name)
{
    internal_.start();
}

ContextedTimer::ScopedContext::~ScopedContext()
{
    internal_.stop();
    parent_.recordContextTime(name_, internal_.elapsed());
}

ContextedTimer::ContextedTimer()
{
    lap_timer_.start();
}

ContextedTimer::ScopedContext ContextedTimer::context(const std::string& name)
{
    return ScopedContext(*this, name);
}

void ContextedTimer::recordContextTime(const std::string& name, double time)
{
    all_contexts_.insert(name);
    current_contexts_[name] += time;
}

void ContextedTimer::nextLap()
{
    lap_timer_.stop();
    double total_time = lap_timer_.elapsed();
    total_times_.push_back(total_time);

    for (const auto& ctx : all_contexts_) {
        double t = 0.0;
        auto it = current_contexts_.find(ctx);
        if (it != current_contexts_.end())
            t = it->second;
        per_lap_contexts_[ctx].push_back(t);
    }

    current_contexts_.clear();
    lap_timer_.reset();
    lap_timer_.start();
}

void ContextedTimer::reset()
{
    lap_timer_.reset();
    lap_timer_.start();
    per_lap_contexts_.clear();
    all_contexts_.clear();
    current_contexts_.clear();
    total_times_.clear();
}

const std::vector<double>& ContextedTimer::totalTimes() const
{
    return total_times_;
}

const std::unordered_map<std::string, std::vector<double>>& ContextedTimer::contextTimes() const
{
    return per_lap_contexts_;
}

std::string ContextedTimer::output(bool per_iteration) const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    if (total_times_.empty()) {
        oss << "No timing data collected.\n";
        return oss.str();
    }

    size_t iters = total_times_.size();
    double total_sum = 0.0;
    for (double t : total_times_)
        total_sum += t;
    double total_avg_s = total_sum / iters;
    double total_avg_ms = total_avg_s * 1000.0;

    oss << "=== TIMING RESULTS ===\n";
    oss << "Iterations: " << iters << "\n";
    oss << "Total avg time: " << total_avg_ms << " ms\n\n";

    if (per_lap_contexts_.empty())
        return oss.str();

    struct Stat { double avg, min, max; };
    std::vector<std::pair<std::string, Stat>> sorted_stats;
    sorted_stats.reserve(per_lap_contexts_.size());

    for (const auto& kv : per_lap_contexts_) {
        const auto& vec = kv.second;
        double sum = 0.0, minv = vec[0], maxv = vec[0];
        for (double v : vec) {
            sum += v;
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        double avg = sum / vec.size();
        sorted_stats.push_back({kv.first, {avg * 1000.0, minv * 1000.0, maxv * 1000.0}});
    }

    std::sort(sorted_stats.begin(), sorted_stats.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    oss << "Context breakdown (average per iteration):\n";
    oss << std::string(74, '-') << "\n";
    oss << std::setw(32) << std::left << "Context"
        << std::setw(12) << std::right << "Avg (ms)"
        << std::setw(12) << "Min"
        << std::setw(12) << "Max"
        << std::setw(8) << "%" << "\n";
    oss << std::string(74, '-') << "\n";

    for (const auto& kv : sorted_stats) {
        const auto& st = kv.second;
        double pct = (total_avg_ms > 0) ? (st.avg / total_avg_ms * 100.0) : 0.0;
        oss << std::setw(32) << std::left << kv.first
            << std::setw(12) << std::right << st.avg
            << std::setw(12) << st.min
            << std::setw(12) << st.max
            << std::setw(7) << std::right << std::setprecision(1) << pct << "%\n";
    }
    oss << std::string(74, '-') << "\n";

    if (per_iteration) {
        oss << "\nPer-iteration total times:\n";
        for (size_t i = 0; i < total_times_.size(); ++i)
            oss << "Lap " << std::setw(3) << i + 1 << ": "
                << total_times_[i] * 1000.0 << " ms\n";
    }

    return oss.str();
}
