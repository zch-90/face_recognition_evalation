#pragma once
#include <thread>
#include <mutex>
#include <list>
#include <memory>
namespace std {
  class thread_group {
  private:
    thread_group(thread_group const&) {}
    thread_group& operator=(thread_group const&) {};
  public:
    thread_group() {}
    ~thread_group() {
      for (auto it = threads.begin(), end = threads.end(); it != end; ++it) {
        delete *it;
      }
    }

    template<typename F>
    thread* create_thread(F threadfunc) {
      lock_guard<mutex> guard(m);
      auto_ptr<thread> new_thread(new thread(threadfunc));
      threads.push_back(new_thread.get());
      return new_thread.release();
    }

    void add_thread(thread* thrd) {
      if (thrd) {
        lock_guard<mutex> guard(m);
        threads.push_back(thrd);
      }
    }

    void remove_thread(thread* thrd) {
      lock_guard<mutex> guard(m);
      auto it = std::find(threads.begin(), threads.end(), thrd);
      if (it != threads.end()) {
        threads.erase(it);
      }
    }

    void join_all() {
      lock_guard<mutex> guard(m);
      for (auto it = threads.begin(), end = threads.end(); it != end; ++it) {
        (*it)->join();
      }
    }

    size_t size() const {
      lock_guard<mutex> guard(m);
      return threads.size();
    }

  private:
    list<thread*> threads;
    mutable mutex m;
  };
}