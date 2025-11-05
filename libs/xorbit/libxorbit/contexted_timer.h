#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include <libbase/timer.h>

class ContextedTimer {
public:
    class ScopedContext {
    public:
        ScopedContext(ContextedTimer& parent, const std::string& name);
        ~ScopedContext();

    private:
        ContextedTimer& parent_;
        std::string name_;
        timer internal_;
    };

public:
    ContextedTimer();

    ScopedContext context(const std::string& name);
    void nextLap();
    void reset();
    std::string output(bool per_iteration = true) const;

    const std::vector<double>& totalTimes() const;
    const std::unordered_map<std::string, std::vector<double>>& contextTimes() const;

private:
    void recordContextTime(const std::string& name, double time);

private:
    timer lap_timer_;
    std::unordered_map<std::string, std::vector<double>> per_lap_contexts_;
    std::unordered_set<std::string> all_contexts_;
    std::unordered_map<std::string, double> current_contexts_;
    std::vector<double> total_times_;
};