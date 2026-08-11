using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace SEGMENT_LABELING
{
    public class Params
    {
        public string Model { get; set; } = "yolo11m-seg.pt";
        public int Epoch { get; set; } = 100;
        public int Batch { get; set; } = 16;
        public int Patience { get; set; } = 20;
        public float Degree { get; set; } = 0.0f;
        public float LearningRate { get; set; } = 0.01f;
        public bool CosineLearningRate { get; set; } = true;

        public int ImageSize { get; set; } = 640;
        public float Mosaic { get; set; } = 0.0f;
        public int CloseMosaic { get; set; } = 10;
        public int MaskRatio { get; set; } = 2;
        public bool OverlapMask { get; set; } = true;
        public bool Amp { get; set; } = true;
        public int Seed { get; set; } = 42;
        public bool Deterministic { get; set; } = true;
        public string CacheMode { get; set; } = "disk";
        public string Device { get; set; } = string.Empty;
        public int Workers { get; set; } = 4;
        public bool AutoExport { get; set; } = true;

        // CPU ONNX Runtime을 사용할 경우 false가 안전합니다.
        public bool ExportHalf { get; set; } = false;

        // 0은 Ultralytics 자동 선택, 특정 런타임 호환성이 필요하면 12 등을 지정합니다.
        public int OnnxOpset { get; set; } = 12;
        public bool SimplifyOnnx { get; set; } = true;
        public bool DynamicOnnx { get; set; } = false;
    }

    public sealed class TrainResult
    {
        public bool IsExport { get; internal set; }
        public bool AutoExportFailed { get; internal set; }
        public int ExitCode { get; internal set; }
        public string StandardOutput { get; internal set; } = string.Empty;
        public string StandardError { get; internal set; } = string.Empty;
        public string BestPtPath { get; internal set; } = string.Empty;
        public string OnnxPath { get; internal set; } = string.Empty;
    }

    public class Train
    {
        public event Action<string> OnLogMessage;

        private readonly object executionLock = new object();
        private bool isRunning;

        public bool IsRunning
        {
            get
            {
                lock (executionLock)
                    return isRunning;
            }
        }

        public async Task<TrainResult> TrainAsync(string pythonExePath, string saveDirectory, string dataYamlPath, Params trainParams)
        {
            if (trainParams == null)
                throw new ArgumentNullException(nameof(trainParams));

            EnterExecution();

            try
            {
                ValidatePythonConfiguration(
                    pythonExePath,
                    dataYamlPath,
                    requireDataset: true);

                ValidateTrainParams(trainParams);

                if (string.IsNullOrWhiteSpace(saveDirectory))
                    throw new ArgumentException("학습 결과 저장 경로가 비어 있습니다.", nameof(saveDirectory));

                Directory.CreateDirectory(saveDirectory);

                string runName =
                    "Run_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);

                string modelName = trainParams.Model == null
                    ? string.Empty
                    : trainParams.Model.Trim();

                if (string.IsNullOrWhiteSpace(modelName))
                    throw new InvalidOperationException(
                        "학습 모델이 비어 있습니다. MODEL 항목에서 yolo11m-seg.pt를 선택하세요.");

                string arguments = string.Join(" ", new[]
                {
                    "--mode", "train",
                    "--model", QuoteArgument(modelName),
                    "--data", QuoteArgument(dataYamlPath),
                    "--epochs", trainParams.Epoch.ToString(CultureInfo.InvariantCulture),
                    "--batch", trainParams.Batch.ToString(CultureInfo.InvariantCulture),
                    "--degrees", trainParams.Degree.ToString(CultureInfo.InvariantCulture),
                    "--lr0", trainParams.LearningRate.ToString(CultureInfo.InvariantCulture),
                    "--patience", trainParams.Patience.ToString(CultureInfo.InvariantCulture),
                    "--cos_lr", ToPythonBool(trainParams.CosineLearningRate),
                    "--imgsz", trainParams.ImageSize.ToString(CultureInfo.InvariantCulture),
                    "--mosaic", trainParams.Mosaic.ToString(CultureInfo.InvariantCulture),
                    "--close_mosaic", trainParams.CloseMosaic.ToString(CultureInfo.InvariantCulture),
                    "--mask_ratio", trainParams.MaskRatio.ToString(CultureInfo.InvariantCulture),
                    "--overlap_mask", ToPythonBool(trainParams.OverlapMask),
                    "--amp", ToPythonBool(trainParams.Amp),
                    "--seed", trainParams.Seed.ToString(CultureInfo.InvariantCulture),
                    "--deterministic", ToPythonBool(trainParams.Deterministic),
                    "--cache", QuoteArgument(trainParams.CacheMode),
                    "--device", QuoteArgument(trainParams.Device),
                    "--workers", trainParams.Workers.ToString(CultureInfo.InvariantCulture),
                    "--auto_export", ToPythonBool(trainParams.AutoExport),
                    "--export_half", ToPythonBool(trainParams.ExportHalf),
                    "--opset", trainParams.OnnxOpset.ToString(CultureInfo.InvariantCulture),
                    "--simplify", ToPythonBool(trainParams.SimplifyOnnx),
                    "--dynamic", ToPythonBool(trainParams.DynamicOnnx),
                    "--save_dir", QuoteArgument(saveDirectory),
                    "--run_name", QuoteArgument(runName)
                });

                return await RunPythonProcessAsync(
                    pythonExePath,
                    arguments,
                    isExport: false);
            }
            finally
            {
                ExitExecution();
            }
        }

        public async Task<TrainResult> ExportOnnxAsync(string pythonExePath, string bestPtPath, Params trainParams)
        {
            if (trainParams == null)
                throw new ArgumentNullException(nameof(trainParams));

            EnterExecution();

            try
            {
                ValidatePythonConfiguration(
                    pythonExePath,
                    dataYamlPath: string.Empty,
                    requireDataset: false);

                if (string.IsNullOrWhiteSpace(bestPtPath) || !File.Exists(bestPtPath))
                    throw new FileNotFoundException("변환할 best.pt 파일을 찾을 수 없습니다.", bestPtPath);

                ValidateExportParams(trainParams);

                string arguments = string.Join(" ", new[]
                {
                    "--mode", "export",
                    "--model", QuoteArgument(bestPtPath),
                    "--imgsz", trainParams.ImageSize.ToString(CultureInfo.InvariantCulture),
                    "--device", QuoteArgument(trainParams.Device),
                    "--export_half", ToPythonBool(trainParams.ExportHalf),
                    "--opset", trainParams.OnnxOpset.ToString(CultureInfo.InvariantCulture),
                    "--simplify", ToPythonBool(trainParams.SimplifyOnnx),
                    "--dynamic", ToPythonBool(trainParams.DynamicOnnx)
                });

                return await RunPythonProcessAsync(
                    pythonExePath,
                    arguments,
                    isExport: true);
            }
            finally
            {
                ExitExecution();
            }
        }

        private void EnterExecution()
        {
            lock (executionLock)
            {
                if (isRunning)
                    throw new InvalidOperationException("이미 학습 또는 ONNX 변환이 실행 중입니다.");

                isRunning = true;
            }
        }

        private void ExitExecution()
        {
            lock (executionLock)
                isRunning = false;
        }

        private static void ValidatePythonConfiguration(
            string pythonExePath,
            string dataYamlPath,
            bool requireDataset)
        {
            if (string.IsNullOrWhiteSpace(pythonExePath))
                throw new InvalidOperationException("Python 실행 경로가 설정되지 않았습니다.");

            // python 또는 python.exe처럼 PATH에서 찾는 명령은 허용합니다.
            bool isPathCommand =
                pythonExePath.IndexOf(Path.DirectorySeparatorChar) < 0 &&
                pythonExePath.IndexOf(Path.AltDirectorySeparatorChar) < 0;

            if (!isPathCommand && !File.Exists(pythonExePath))
                throw new FileNotFoundException("Python 인터프리터를 찾을 수 없습니다.", pythonExePath);

            if (requireDataset &&
                (string.IsNullOrWhiteSpace(dataYamlPath) || !File.Exists(dataYamlPath)))
            {
                throw new FileNotFoundException(
                    "먼저 [LABELS to YAML] 버튼으로 data.yaml을 생성하세요.",
                    dataYamlPath);
            }
        }

        private static void ValidateRequiredCommandArguments(
            string arguments,
            bool isExport)
        {
            if (string.IsNullOrWhiteSpace(arguments))
                throw new InvalidOperationException("Python 실행 인자가 비어 있습니다.");

            if (arguments.IndexOf("--model", StringComparison.OrdinalIgnoreCase) < 0)
            {
                throw new InvalidOperationException(
                    "Python 실행 인자에 --model이 없습니다. " +
                    "Train.cs와 Form1.cs가 동일한 수정본인지 확인하세요.");
            }

            string expectedMode = isExport ? "--mode export" : "--mode train";
            if (arguments.IndexOf(expectedMode, StringComparison.OrdinalIgnoreCase) < 0)
            {
                throw new InvalidOperationException(
                    "Python 실행 모드가 올바르지 않습니다. 예상값: " + expectedMode);
            }
        }

        private static void ValidateTrainParams(Params value)
        {
            if (string.IsNullOrWhiteSpace(value.Model))
                throw new ArgumentException("학습 모델이 선택되지 않았습니다.");

            if (value.Epoch < 1)
                throw new ArgumentOutOfRangeException(nameof(value.Epoch), "Epoch는 1 이상이어야 합니다.");

            if (value.Batch == 0 || value.Batch < -1)
                throw new ArgumentOutOfRangeException(nameof(value.Batch), "Batch는 -1 또는 1 이상이어야 합니다.");

            if (value.Degree < 0.0f || value.Degree > 180.0f)
                throw new ArgumentOutOfRangeException(nameof(value.Degree), "Degree는 0~180 범위여야 합니다.");

            if (value.LearningRate <= 0.0f || value.LearningRate > 1.0f)
                throw new ArgumentOutOfRangeException(
                    nameof(value.LearningRate),
                    "Learning Rate는 0보다 크고 1 이하여야 합니다.");

            if (value.Patience < 0)
                throw new ArgumentOutOfRangeException(nameof(value.Patience), "Patience는 0 이상이어야 합니다.");

            if (value.ImageSize < 1)
                throw new ArgumentOutOfRangeException(nameof(value.ImageSize), "Image Size는 1 이상이어야 합니다.");

            if (value.Mosaic < 0.0f || value.Mosaic > 1.0f)
                throw new ArgumentOutOfRangeException(nameof(value.Mosaic), "Mosaic은 0~1 범위여야 합니다.");

            if (value.CloseMosaic < 0)
                throw new ArgumentOutOfRangeException(nameof(value.CloseMosaic), "Close Mosaic은 0 이상이어야 합니다.");

            if (value.MaskRatio < 1)
                throw new ArgumentOutOfRangeException(nameof(value.MaskRatio), "Mask Ratio는 1 이상이어야 합니다.");

            if (value.Workers < 0)
                throw new ArgumentOutOfRangeException(nameof(value.Workers), "Workers는 0 이상이어야 합니다.");

            if (value.Seed < 0)
                throw new ArgumentOutOfRangeException(nameof(value.Seed), "Seed는 0 이상이어야 합니다.");

            if (!string.Equals(value.CacheMode, "none", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(value.CacheMode, "ram", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(value.CacheMode, "disk", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException("CacheMode는 none, ram, disk 중 하나여야 합니다.");
            }

            ValidateExportParams(value);
        }

        private static void ValidateExportParams(Params value)
        {
            if (value.ImageSize < 1)
                throw new ArgumentOutOfRangeException(nameof(value.ImageSize), "Image Size는 1 이상이어야 합니다.");

            if (value.OnnxOpset < 0)
                throw new ArgumentOutOfRangeException(nameof(value.OnnxOpset), "ONNX Opset은 0 이상이어야 합니다.");
        }

        private async Task<TrainResult> RunPythonProcessAsync(
            string pythonExePath,
            string arguments,
            bool isExport)
        {
            ValidateRequiredCommandArguments(arguments, isExport);

            Log("Python 실행 파일: " + pythonExePath);
            Log("Python 실행 인자: -u " + arguments);

            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = pythonExePath,
                Arguments = arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };

            start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            start.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

            Log(isExport ? "ONNX 변환을 시작합니다." : "모델 학습을 시작합니다.");

            ProcessResult processResult =
                await Task.Run(() => ExecuteProcess(start));

            TrainResult result = new TrainResult
            {
                IsExport = isExport,
                ExitCode = processResult.ExitCode,
                StandardOutput = processResult.StandardOutput,
                StandardError = processResult.StandardError,
                BestPtPath = ExtractMarker(processResult.StandardOutput, "RESULT_BEST_PT="),
                OnnxPath = ExtractMarker(processResult.StandardOutput, "RESULT_ONNX="),
                AutoExportFailed =
                    processResult.StandardError.IndexOf(
                        "AUTO_EXPORT_FAILED=",
                        StringComparison.OrdinalIgnoreCase) >= 0
            };

            if (processResult.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    "Python 프로세스가 비정상 종료되었습니다. " +
                    "(ExitCode: " + processResult.ExitCode + ")\n\n" +
                    BuildLogSummary(processResult.StandardError, processResult.StandardOutput));
            }

            if (!isExport && result.AutoExportFailed)
            {
                Log(
                    "모델 학습은 완료되었지만 ONNX 자동 변환에 실패했습니다.\n" +
                    BuildLogSummary(processResult.StandardError, processResult.StandardOutput));
            }
            else
            {
                Log(isExport ? "ONNX 변환이 완료되었습니다." : "모델 학습이 완료되었습니다.");
            }

            return result;
        }

        private ProcessResult ExecuteProcess(ProcessStartInfo start)
        {
            StringBuilder output = new StringBuilder();
            StringBuilder error = new StringBuilder();
            object outputLock = new object();
            object errorLock = new object();

            using (Process process = new Process())
            {
                process.StartInfo = start;
                process.EnableRaisingEvents = true;

                process.OutputDataReceived += delegate (object sender, DataReceivedEventArgs e)
                {
                    if (string.IsNullOrEmpty(e.Data))
                        return;

                    lock (outputLock)
                        output.AppendLine(e.Data);

                    Log(e.Data);
                    Debug.WriteLine(e.Data);
                };

                process.ErrorDataReceived += delegate (object sender, DataReceivedEventArgs e)
                {
                    if (string.IsNullOrEmpty(e.Data))
                        return;

                    lock (errorLock)
                        error.AppendLine(e.Data);

                    Log("[Python] " + e.Data);
                    Debug.WriteLine("[Python Error] " + e.Data);
                };

                if (!process.Start())
                    throw new InvalidOperationException("Python 프로세스를 시작하지 못했습니다.");

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                process.WaitForExit();

                return new ProcessResult
                {
                    ExitCode = process.ExitCode,
                    StandardOutput = output.ToString(),
                    StandardError = error.ToString()
                };
            }
        }

        private void Log(string message)
        {
            Action<string> handler = OnLogMessage;
            if (handler != null)
                handler("[ObjectSegmentation] " + message);
        }

        private static string ExtractMarker(string text, string marker)
        {
            if (string.IsNullOrWhiteSpace(text))
                return string.Empty;

            string[] lines = text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);

            for (int i = lines.Length - 1; i >= 0; i--)
            {
                if (lines[i].StartsWith(marker, StringComparison.OrdinalIgnoreCase))
                    return lines[i].Substring(marker.Length).Trim();
            }

            return string.Empty;
        }

        private static string BuildLogSummary(string error, string output)
        {
            StringBuilder combined = new StringBuilder();

            if (!string.IsNullOrWhiteSpace(error))
            {
                combined.AppendLine("[Standard Error]");
                combined.AppendLine(error.Trim());
            }

            if (!string.IsNullOrWhiteSpace(output))
            {
                combined.AppendLine("[Standard Output]");
                combined.AppendLine(output.Trim());
            }

            if (combined.Length == 0)
                return "상세 로그가 없습니다.";

            const int maxLength = 6000;
            string value = combined.ToString();

            return value.Length <= maxLength
                ? value
                : value.Substring(value.Length - maxLength, maxLength);
        }

        private static string ToPythonBool(bool value)
        {
            return value ? "true" : "false";
        }

        private static string QuoteArgument(string value)
        {
            value = value ?? string.Empty;

            StringBuilder quoted = new StringBuilder();
            quoted.Append('"');

            int backslashCount = 0;

            foreach (char character in value)
            {
                if (character == '\\')
                {
                    backslashCount++;
                    continue;
                }

                if (character == '"')
                {
                    quoted.Append('\\', backslashCount * 2 + 1);
                    quoted.Append('"');
                    backslashCount = 0;
                    continue;
                }

                if (backslashCount > 0)
                {
                    quoted.Append('\\', backslashCount);
                    backslashCount = 0;
                }

                quoted.Append(character);
            }

            // 닫는 따옴표 앞의 역슬래시는 두 배로 처리합니다.
            if (backslashCount > 0)
                quoted.Append('\\', backslashCount * 2);

            quoted.Append('"');
            return quoted.ToString();
        }
    }

    internal sealed class ProcessResult
    {
        public int ExitCode { get; set; }
        public string StandardOutput { get; set; } = string.Empty;
        public string StandardError { get; set; } = string.Empty;
    }
}
