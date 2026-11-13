#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace GRIM { namespace Perception { namespace Physical {

// One-slot worker used by main-loop integration points whose real work may
// take longer than a render frame. RequestLatest() never waits: requests made
// while work is running collapse into one follow-up invocation, and that
// invocation is expected to pull the newest available bus value.
class PhysicalLatestTickWorker {
public:
    PhysicalLatestTickWorker() = default;
    ~PhysicalLatestTickWorker() { Stop(); }

    PhysicalLatestTickWorker(const PhysicalLatestTickWorker&) = delete;
    PhysicalLatestTickWorker& operator=(const PhysicalLatestTickWorker&) = delete;

    void Start(std::function<void()> work) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (started_) return;
        work_ = std::move(work);
        stop_requested_ = false;
        work_pending_ = false;
        started_ = true;
        thread_ = std::thread([this] { WorkerMain(); });
    }

    void RequestLatest() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (!started_ || stop_requested_) return;
            work_pending_ = true;
        }
        cv_.notify_one();
    }

    void Stop() {
        std::thread thread_to_join;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (!started_) return;
            stop_requested_ = true;
            work_pending_ = false;
            thread_to_join = std::move(thread_);
            started_ = false;
        }
        cv_.notify_all();
        if (thread_to_join.joinable()) thread_to_join.join();
        std::lock_guard<std::mutex> lk(mutex_);
        work_ = {};
        stop_requested_ = false;
    }

    bool IsStarted() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return started_ && !stop_requested_;
    }

private:
    void WorkerMain() {
        for (;;) {
            std::function<void()> work;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_.wait(lk, [this] { return stop_requested_ || work_pending_; });
                if (stop_requested_) return;
                work_pending_ = false;
                work = work_;
            }
            if (work) work();
        }
    }

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::thread             thread_;
    std::function<void()>   work_;
    bool                    started_ = false;
    bool                    stop_requested_ = false;
    bool                    work_pending_ = false;
};

}}} // namespace GRIM::Perception::Physical
