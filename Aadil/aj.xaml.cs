using System;
using System.IO;
using System.Windows;
using System.Diagnostics;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using LiveCharts;
using LiveCharts.Wpf;
using System.Threading.Tasks;

namespace WpfApp2
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private ChartValues<double> _chartValues;
        private string _processorName;
        private string _operatingSystem;
        private double _singleCoreScore;
        private double _multiCoreScore;
        private bool _isBenchmarkRunning;

        private readonly string BenchmarkPath = @"C:\Users\mehar\source\repos\WpfApp2\WpfApp2\bin\Debug\net6.0-windows\CPUBenchmark.exe.exe";
        private readonly PerformanceCounter cpuCounter;

        public ChartValues<double> ChartValues
        {
            get => _chartValues;
            set
            {
                _chartValues = value;
                OnPropertyChanged();
            }
        }

        public string ProcessorName
        {
            get => _processorName;
            set
            {
                _processorName = value;
                OnPropertyChanged();
            }
        }

        public string OperatingSystem
        {
            get => _operatingSystem;
            set
            {
                _operatingSystem = value;
                OnPropertyChanged();
            }
        }

        public double SingleCoreScore
        {
            get => _singleCoreScore;
            set
            {
                _singleCoreScore = value;
                OnPropertyChanged();
            }
        }

        public double MultiCoreScore
        {
            get => _multiCoreScore;
            set
            {
                _multiCoreScore = value;
                OnPropertyChanged();
            }
        }

        public MainWindow()
        {
            InitializeComponent();
            ChartValues = new ChartValues<double>();
            DataContext = this;
            GetSystemInfo();

            try
            {
                cpuCounter = new PerformanceCounter("Processor", "% Processor Time", "_Total");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error initializing performance counter: {ex.Message}");
            }
        }

        private void GetSystemInfo()
        {
            try
            {
                ProcessorName = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "Unknown Processor";
                OperatingSystem = Environment.OSVersion.ToString();
                OperatingSystem += Environment.Is64BitOperatingSystem ? " 64-bit" : " 32-bit";
            }
            catch (Exception ex)
            {
                ProcessorName = "Unable to detect";
                OperatingSystem = "Unable to detect";
                MessageBox.Show($"Error getting system info: {ex.Message}", "System Info Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private async void StartBenchmark_Click(object sender, RoutedEventArgs e)
        {
            if (_isBenchmarkRunning) return;

            _isBenchmarkRunning = true;
            StartButton.IsEnabled = false;
            ChartValues.Clear();

            try
            {
                if (!File.Exists(BenchmarkPath))
                {
                    MessageBox.Show($"Error: CPUBenchmark.exe not found at:\n{BenchmarkPath}", "File Not Found", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                var monitoringTask = MonitorCPUUsageAsync();

                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = BenchmarkPath,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(BenchmarkPath)
                    }
                };

                process.OutputDataReceived += (s, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        ParseBenchmarkOutput(args.Data);
                    }
                };

                process.Exited += (s, args) =>
                {
                    _isBenchmarkRunning = false;
                    Application.Current.Dispatcher.Invoke(() => StartButton.IsEnabled = true);
                };

                process.EnableRaisingEvents = true;
                process.Start();
                process.BeginOutputReadLine();

                await process.WaitForExitAsync();

                _isBenchmarkRunning = false;
                await monitoringTask;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error running benchmark: {ex.Message}", "Execution Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                _isBenchmarkRunning = false;
                StartButton.IsEnabled = true;
            }
        }

        private async Task MonitorCPUUsageAsync()
        {
            if (cpuCounter == null) return;

            while (_isBenchmarkRunning)
            {
                try
                {
                    var cpuUsage = cpuCounter.NextValue();
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        ChartValues.Add(cpuUsage);
                    });
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error monitoring CPU: {ex.Message}");
                }

                await Task.Delay(1000);
            }
        }

        private void ParseBenchmarkOutput(string output)
        {
            try
            {
                if (output.Contains("Single:"))
                {
                    var score = ParseScore(output, "Single:");
                    Application.Current.Dispatcher.Invoke(() => SingleCoreScore = score);
                }
                else if (output.Contains("Multi:"))
                {
                    var score = ParseScore(output, "Multi:");
                    Application.Current.Dispatcher.Invoke(() => MultiCoreScore = score);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error parsing benchmark output: {ex.Message}");
            }
        }

        private double ParseScore(string output, string prefix)
        {
            var scoreStr = output.Substring(output.IndexOf(prefix) + prefix.Length).Trim();
            return double.TryParse(scoreStr, out double score) ? score : 0;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}