from programBenchmarker import InfiniteBenchmarker, FiniteBenchmarker



benchmarker = FiniteBenchmarker("python finiteTestfile.py")
benchmarker.run()
benchmarker.report()


benchmarker = InfiniteBenchmarker("python infiniteTestfile.py")
benchmarker.run(duration=600)
benchmarker.report()

