// Verify the exported policy independently of Gazebo and optionally replay RL traces.
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iostream>
#include <array>
#include <cmath>
int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: policy_contract_test MODEL [TRACE_TEXT]\n"; return 2; }
    try {
        at::set_num_threads(1);
        torch::NoGradGuard guard;
        auto model = torch::jit::load(argv[1], torch::kCPU);
        model.eval();
        if (argc == 2) {
            for (int i=0; i<100; ++i) {
                auto action = model.get_method("act_inference")({torch::zeros({1,225})}).toTensor();
                if (action.numel() != 12 || !torch::isfinite(action).all().item<bool>()) return 1;
            }
            std::cout << "PASS: 100 CPU inferences, [1,225] -> 12 finite actions\n";
        } else {
            std::ifstream input(argv[2]);
            if (!input) return 2;
            std::array<float,225> observation{};
            std::array<float,12> expected{};
            float stamp;
            int count=0; double max_error=0;
            while (input >> stamp) {
                for (auto& v: observation) if (!(input >> v)) return 2;
                for (auto& v: expected) if (!(input >> v)) return 2;
                auto action = model.get_method("act_inference")({torch::from_blob(observation.data(),{1,225},torch::kFloat32).clone()})
                    .toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
                for (int i=0; i<12; ++i) max_error=std::max(max_error,std::abs(double(action.data_ptr<float>()[i]-expected[i])));
                ++count;
            }
            std::cout << "Replayed " << count << " traces; max action error=" << max_error << '\n';
            if (count == 0 || !std::isfinite(max_error) || max_error > 1e-4) return 1;
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
