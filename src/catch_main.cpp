#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>
#include <iomanip>

class TabularReporter : public Catch::StreamingReporterBase {
public:
    using StreamingReporterBase::StreamingReporterBase;

    static std::string getDescription() {
        return "Tabular Reporter for GPU Benchmarks";
    }

    void testRunStarting(Catch::TestRunInfo const& _testRunInfo ) override {
        std::cout << "# Times (ns) for                   "  
		  << "estimate  iter  samples resamp clckRes  ovrhd "
		  << " mean        std       m_lo       m_hi      "
		  << " std_lo     std_hi    outVar  " << "\n";
    }

    void testCasePartialStarting(Catch::TestCaseInfo const& testInfo,
                                 uint64_t partNumber) override {
        std::cout << std::setw(32) << testInfo.name << " ";
    }

    void benchmarkPreparing(Catch::StringRef stringRef ) override {
        //std::cout << "benchmarkPreparing " << '\n';
    }

    void benchmarkStarting(Catch::BenchmarkInfo const& benchmarkInfo ) override {
        std::cout << std::setw(10) << benchmarkInfo.estimatedDuration << " " 
		  << std::setw(4) << benchmarkInfo.iterations << " " 
		  << std::setw(6) << benchmarkInfo.samples << " " 
		  << std::setw(8) << benchmarkInfo.resamples << " " 
		  << std::setw(8) << benchmarkInfo.clockResolution << " " 
		  << std::setw(4) << benchmarkInfo.clockCost  << " ";
    }

    void benchmarkEnded(Catch::BenchmarkStats<> const& benchmarkStats ) override {
        std::cout << std::setw(10) << benchmarkStats.mean.point.count() << " " 
		  << std::setw(10) << benchmarkStats.standardDeviation.point.count() << " " 
		  << std::setw(10) << benchmarkStats.mean.lower_bound.count() << " " 
		  << std::setw(10) << benchmarkStats.mean.upper_bound.count() << " " 
		  << std::setw(10) << benchmarkStats.standardDeviation.lower_bound.count() << " " 
		  << std::setw(10) << benchmarkStats.standardDeviation.upper_bound.count() << " " 
		  << std::setw(10) << benchmarkStats.outlierVariance << '\n';
    }

    void benchmarkFailed(Catch::StringRef stringRef ) override {
        //std::cout << "benchmarkFailed " << '\n';
    }
};

CATCH_REGISTER_REPORTER("tabular", TabularReporter)
