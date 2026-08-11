using OpenCvSharp;
using Seojin.RobotVision;
using System;
using System.Linq;

namespace SEGMENT_LABELING
{
    /// <summary>
    /// 평면 사각형 객체의 카메라 기준 6DoF 자세 결과입니다.
    /// TranslationMm: 카메라 좌표계 X/Y/Z [mm]
    /// EulerDegrees: Roll(X), Pitch(Y), Yaw(Z) [deg]
    /// </summary>
    public sealed class Pose3DResult
    {
        public double[] RotationVector { get; internal set; } = new double[3];
        public double[] TranslationMm { get; internal set; } = new double[3];
        public double[] EulerDegrees { get; internal set; } = new double[3];
        public double[,] RotationMatrix { get; internal set; } = new double[3, 3];
        public double[,] ObjectToCameraMatrix { get; internal set; } = new double[4, 4];
        public double ReprojectionErrorPx { get; internal set; }
        public SolvePnPFlags SolverUsed { get; internal set; }
        public bool UsedIterativeFallback { get; internal set; }

        public double Xmm => TranslationMm[0];
        public double Ymm => TranslationMm[1];
        public double Zmm => TranslationMm[2];
        public double RollDeg => EulerDegrees[0];
        public double PitchDeg => EulerDegrees[1];
        public double YawDeg => EulerDegrees[2];
    }


    /// <summary>
    /// MinAreaRect 기반 XYθ 정렬 결과입니다.
    /// X/Y는 원본 영상 픽셀 좌표이고, Theta는 객체 장축의 회전각입니다.
    /// </summary>
    public sealed class XYThetaAlignResult
    {
        public double CenterXPixel { get; internal set; }
        public double CenterYPixel { get; internal set; }
        public double ThetaDeg { get; internal set; }
        public double MajorLengthPixel { get; internal set; }
        public double MinorLengthPixel { get; internal set; }
    }

    /// <summary>
    /// MinAreaRect 기반 XYθ 및 카메라 캘리브레이션 기반 6DoF 계산 도구입니다.
    /// </summary>
    public static class Align3D
    {

        /// <summary>
        /// Instance Segmentation Contour의 MinAreaRect에서 중심 X/Y와 장축 회전각을 계산합니다.
        /// 반환 각도 범위는 -90도 이상 90도 미만입니다.
        ///
        /// [중요/한계] MinAreaRect는 대칭 사각형이므로 각도는 180° 주기로만 정의됩니다.
        /// 즉 이 ThetaDeg만으로는 대상이 정방향인지 180° 뒤집힌 상태인지 구분할 수 없습니다.
        /// 대상의 앞/뒤(정방향)를 반드시 구분해야 하는 정렬이라면, 별도의 비대칭
        /// Keypoint 모델이나 방향 마커(예: 한쪽 모서리의 노치, 인쇄 마크)로 얻은
        /// 기준점을 <see cref="ResolveFullRotationDeg"/>에 함께 전달해 (-180,180] 범위의
        /// 실제 방향각으로 보정해야 합니다.
        /// </summary>
        public static XYThetaAlignResult EstimateXYTheta(RotatedRect rotatedRect)
        {
            if (rotatedRect.Size.Width <= 0.0f ||
                rotatedRect.Size.Height <= 0.0f)
            {
                throw new ArgumentException(
                    "유효한 MinAreaRect가 필요합니다.",
                    nameof(rotatedRect));
            }

            double angle = rotatedRect.Angle;
            double majorLength;
            double minorLength;

            if (rotatedRect.Size.Width >= rotatedRect.Size.Height)
            {
                majorLength = rotatedRect.Size.Width;
                minorLength = rotatedRect.Size.Height;
            }
            else
            {
                // OpenCV RotatedRect의 Width 방향에서 Height 방향(장축)으로 90도 회전합니다.
                angle += 90.0;
                majorLength = rotatedRect.Size.Height;
                minorLength = rotatedRect.Size.Width;
            }

            angle = NormalizeSigned90(angle);

            return new XYThetaAlignResult
            {
                CenterXPixel = rotatedRect.Center.X,
                CenterYPixel = rotatedRect.Center.Y,
                ThetaDeg = angle,
                MajorLengthPixel = majorLength,
                MinorLengthPixel = minorLength
            };
        }

        /// <summary>
        /// EstimateXYTheta가 반환하는 (-90,90] 각도는 MinAreaRect의 180° 대칭성 때문에
        /// 대상의 실제 정방향(0~360°)을 특정하지 못합니다. 비대칭 마커(Keypoint, 노치 등)의
        /// 픽셀 좌표 하나를 함께 넘기면, 그 마커가 장축의 +방향과 -방향 중 어느 쪽에
        /// 있는지로 실제 방향각을 (-180,180] 범위로 복원합니다.
        /// </summary>
        /// <param name="xyTheta">EstimateXYTheta의 결과.</param>
        /// <param name="frontMarkerPixel">
        /// 대상의 "정방향"을 나타내는 비대칭 기준점의 영상 픽셀 좌표
        /// (Keypoint 모델 출력 또는 방향 마커 검출 결과).
        /// </param>
        public static double ResolveFullRotationDeg(
            XYThetaAlignResult xyTheta,
            Point2f frontMarkerPixel)
        {
            if (xyTheta == null)
                throw new ArgumentNullException(nameof(xyTheta));

            double thetaRad = xyTheta.ThetaDeg * Math.PI / 180.0;
            double axisX = Math.Cos(thetaRad);
            double axisY = Math.Sin(thetaRad);

            double markerVecX = frontMarkerPixel.X - xyTheta.CenterXPixel;
            double markerVecY = frontMarkerPixel.Y - xyTheta.CenterYPixel;

            // 마커가 장축의 +방향 쪽에 있으면 ThetaDeg 그대로, -방향 쪽에 있으면
            // 180°를 더해 실제 정방향으로 뒤집습니다.
            double dot = (markerVecX * axisX) + (markerVecY * axisY);
            double fullAngle = dot >= 0.0
                ? xyTheta.ThetaDeg
                : xyTheta.ThetaDeg + 180.0;

            while (fullAngle > 180.0) fullAngle -= 360.0;
            while (fullAngle <= -180.0) fullAngle += 360.0;

            return fullAngle;
        }

        /// <summary>
        /// 카메라 내부 파라미터 행렬을 생성합니다.
        /// </summary>
        public static double[,] CreateCameraMatrix(
            double fx,
            double fy,
            double cx,
            double cy)
        {
            if (fx <= 0.0 || fy <= 0.0)
                throw new ArgumentOutOfRangeException("fx/fy는 0보다 커야 합니다.");

            return new double[,]
            {
                { fx, 0.0, cx },
                { 0.0, fy, cy },
                { 0.0, 0.0, 1.0 }
            };
        }

        /// <summary>
        /// 평면 사각형 객체의 6DoF 자세를 계산합니다.
        /// imageCorners 순서는 TL, TR, BR, BL입니다.
        /// objectWidthMm/objectHeightMm는 실제 대상의 외곽 치수입니다.
        /// </summary>
        public static Pose3DResult EstimatePose(
            Point2f[] imageCorners,
            double objectWidthMm,
            double objectHeightMm,
            double[,] cameraMatrix,
            double[] distortionCoefficients)
        {
            if (imageCorners == null || imageCorners.Length != 4)
                throw new ArgumentException("네 모서리 좌표가 필요합니다.", nameof(imageCorners));

            if (objectWidthMm <= 0.0 || objectHeightMm <= 0.0)
                throw new ArgumentOutOfRangeException("실물 폭과 높이는 0보다 커야 합니다.");

            ValidateCameraMatrix(cameraMatrix);

            Point2f[] orderedCorners =
                InstanceSegmentation.OrderCorners(imageCorners);

            if (orderedCorners.Length != 4)
                throw new InvalidOperationException("모서리 정렬에 실패했습니다.");

            float halfWidth = (float)(objectWidthMm / 2.0);
            float halfHeight = (float)(objectHeightMm / 2.0);

            // 객체 좌표계: X=우측, Y=상측, Z=평면 법선 방향입니다.
            // 순서: TL, TR, BR, BL입니다.
            Point3f[] objectPoints =
            {
                new Point3f(-halfWidth,  halfHeight, 0f),
                new Point3f( halfWidth,  halfHeight, 0f),
                new Point3f( halfWidth, -halfHeight, 0f),
                new Point3f(-halfWidth, -halfHeight, 0f)
            };

            double[] dist = distortionCoefficients ?? Array.Empty<double>();
            double[] rvec;
            double[] tvec;

            // 현재 프로젝트의 OpenCvSharp 버전에는 IPPE/IPPE_SQUARE가 없고,
            // Cv2.SolvePnP 반환 형식도 void이므로 Iterative를 사용합니다.
            // OpenCV의 Iterative 방식은 평면 점에 대해 homography 기반 초기값을 사용합니다.
            SolvePnPFlags primarySolver = SolvePnPFlags.Iterative;

            bool solved = TrySolvePose(
                objectPoints,
                orderedCorners,
                cameraMatrix,
                dist,
                primarySolver,
                out rvec,
                out tvec);

            if (!solved)
                throw new InvalidOperationException("SolvePnP 결과가 유효하지 않습니다.");

            double[,] rotationMatrix = RodriguesToMatrix(rvec);
            double[] euler = RotationMatrixToEulerDegrees(rotationMatrix);
            double error = ComputeReprojectionError(
                objectPoints,
                orderedCorners,
                rvec,
                tvec,
                cameraMatrix,
                dist);

            return new Pose3DResult
            {
                RotationVector = (double[])rvec.Clone(),
                TranslationMm = (double[])tvec.Clone(),
                EulerDegrees = euler,
                RotationMatrix = rotationMatrix,
                ObjectToCameraMatrix = CreateTransformMatrix(rotationMatrix, tvec),
                ReprojectionErrorPx = error,
                SolverUsed = primarySolver,
                UsedIterativeFallback = false
            };
        }

        private static bool TrySolvePose(
            Point3f[] objectPoints,
            Point2f[] imagePoints,
            double[,] cameraMatrix,
            double[] distortionCoefficients,
            SolvePnPFlags solver,
            out double[] rvec,
            out double[] tvec)
        {
            rvec = new double[3];
            tvec = new double[3];

            try
            {
                // 사용 중인 OpenCvSharp 버전에서는 SolvePnP가 void를 반환합니다.
                Cv2.SolvePnP(
                    objectPoints,
                    imagePoints,
                    cameraMatrix,
                    distortionCoefficients,
                    ref rvec,
                    ref tvec,
                    false,
                    solver);

                return IsValidPose(rvec, tvec);
            }
            catch (OpenCvSharpException)
            {
                return false;
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        private static bool IsValidPose(
            double[] rvec,
            double[] tvec)
        {
            if (rvec == null || rvec.Length < 3 ||
                tvec == null || tvec.Length < 3)
            {
                return false;
            }

            if (rvec.Any(v => double.IsNaN(v) || double.IsInfinity(v)) ||
                tvec.Any(v => double.IsNaN(v) || double.IsInfinity(v)))
            {
                return false;
            }

            // 객체는 카메라 전방에 있어야 합니다.
            return tvec[2] > 0.0;
        }

        /// <summary>
        /// 네 모서리를 기준으로 원근을 제거하고 정면 영상으로 펼칩니다.
        /// 이것은 2D homography 정렬이며, 캘리브레이션 없이도 사용할 수 있습니다.
        /// </summary>
        public static Mat RectifyPerspective(
            Mat source,
            Point2f[] imageCorners,
            int outputWidth,
            int outputHeight)
        {
            if (source == null || source.Empty())
                throw new ArgumentException("입력 영상이 비어 있습니다.", nameof(source));

            if (imageCorners == null || imageCorners.Length != 4)
                throw new ArgumentException("네 모서리 좌표가 필요합니다.", nameof(imageCorners));

            if (outputWidth <= 1 || outputHeight <= 1)
                throw new ArgumentOutOfRangeException("출력 크기는 1보다 커야 합니다.");

            Point2f[] ordered = InstanceSegmentation.OrderCorners(imageCorners);
            Point2f[] destination =
            {
                new Point2f(0f, 0f),
                new Point2f(outputWidth - 1f, 0f),
                new Point2f(outputWidth - 1f, outputHeight - 1f),
                new Point2f(0f, outputHeight - 1f)
            };

            Mat aligned = new Mat();

            using (Mat perspective = Cv2.GetPerspectiveTransform(ordered, destination))
            {
                Cv2.WarpPerspective(
                    source,
                    aligned,
                    perspective,
                    new OpenCvSharp.Size(outputWidth, outputHeight),
                    InterpolationFlags.Linear,
                    BorderTypes.Constant,
                    Scalar.Black);
            }

            return aligned;
        }


        /// <summary>
        /// 180도 주기의 축 방향 각도를 [-90, +90) 범위로 정규화합니다.
        /// 검출 각도와 보정 오차가 동일한 경계 규칙을 사용하도록 외부에도 공개합니다.
        /// </summary>
        public static double NormalizeSigned90(double angle)
        {
            if (double.IsNaN(angle) || double.IsInfinity(angle))
                throw new ArgumentOutOfRangeException(nameof(angle), "유한한 각도값이 필요합니다.");

            angle %= 180.0;

            if (angle >= 90.0)
                angle -= 180.0;
            else if (angle < -90.0)
                angle += 180.0;

            return angle;
        }

        private static void ValidateCameraMatrix(double[,] cameraMatrix)
        {
            if (cameraMatrix == null ||
                cameraMatrix.GetLength(0) != 3 ||
                cameraMatrix.GetLength(1) != 3)
            {
                throw new ArgumentException("CameraMatrix는 3x3 배열이어야 합니다.", nameof(cameraMatrix));
            }

            if (cameraMatrix[0, 0] <= 0.0 || cameraMatrix[1, 1] <= 0.0)
                throw new ArgumentException("CameraMatrix의 fx/fy가 올바르지 않습니다.", nameof(cameraMatrix));
        }

        private static double ComputeReprojectionError(
            Point3f[] objectPoints,
            Point2f[] measuredPoints,
            double[] rvec,
            double[] tvec,
            double[,] cameraMatrix,
            double[] distortionCoefficients)
        {
            Cv2.ProjectPoints(
                objectPoints,
                rvec,
                tvec,
                cameraMatrix,
                distortionCoefficients,
                out Point2f[] projectedPoints,
                out double[,] jacobian);

            if (projectedPoints == null || projectedPoints.Length != measuredPoints.Length)
                return double.PositiveInfinity;

            double squaredError = 0.0;

            for (int i = 0; i < projectedPoints.Length; i++)
            {
                double dx = projectedPoints[i].X - measuredPoints[i].X;
                double dy = projectedPoints[i].Y - measuredPoints[i].Y;
                squaredError += (dx * dx) + (dy * dy);
            }

            return Math.Sqrt(squaredError / projectedPoints.Length);
        }

        private static double[,] RodriguesToMatrix(double[] rvec)
        {
            if (rvec == null || rvec.Length < 3)
                throw new ArgumentException("rvec는 길이 3의 배열이어야 합니다.", nameof(rvec));

            double rx = rvec[0];
            double ry = rvec[1];
            double rz = rvec[2];
            double theta = Math.Sqrt((rx * rx) + (ry * ry) + (rz * rz));

            if (theta < 1e-12)
            {
                return new double[,]
                {
                    { 1.0, 0.0, 0.0 },
                    { 0.0, 1.0, 0.0 },
                    { 0.0, 0.0, 1.0 }
                };
            }

            double x = rx / theta;
            double y = ry / theta;
            double z = rz / theta;
            double c = Math.Cos(theta);
            double s = Math.Sin(theta);
            double v = 1.0 - c;

            return new double[,]
            {
                { (x * x * v) + c,       (x * y * v) - (z * s), (x * z * v) + (y * s) },
                { (y * x * v) + (z * s), (y * y * v) + c,       (y * z * v) - (x * s) },
                { (z * x * v) - (y * s), (z * y * v) + (x * s), (z * z * v) + c       }
            };
        }

        /// <summary>
        /// R = Rz(yaw) * Ry(pitch) * Rx(roll) 기준 ZYX Extrinsic Euler 분해입니다.
        /// Gimbal Lock(pitch = ±90°) 발생 시 pitch 부호에 따라 올바른 roll 수식을 분리 적용합니다.
        /// </summary>
        private static double[] RotationMatrixToEulerDegrees(double[,] r)
        {
            double sy = Math.Sqrt((r[0, 0] * r[0, 0]) + (r[1, 0] * r[1, 0]));
            bool singular = sy < 1e-9;

            double roll;
            double pitch;
            double yaw;

            if (!singular)
            {
                roll = Math.Atan2(r[2, 1], r[2, 2]);
                pitch = Math.Atan2(-r[2, 0], sy);
                yaw = Math.Atan2(r[1, 0], r[0, 0]);
            }
            else
            {
                // Gimbal Lock: pitch = ±90°, yaw = 0으로 고정하고 roll만 복원합니다.
                // r[2,0] = -sin(pitch)
                // pitch = +90° (r[2,0] = -1): roll = Atan2( r[0,1],  r[0,2])
                // pitch = -90° (r[2,0] = +1): roll = Atan2(-r[0,1], -r[0,2])
                if (r[2, 0] < 0.0)
                {
                    pitch = Math.PI / 2.0;
                    roll = Math.Atan2(r[0, 1], r[0, 2]);
                }
                else
                {
                    pitch = -Math.PI / 2.0;
                    roll = Math.Atan2(-r[0, 1], -r[0, 2]);
                }
                yaw = 0.0;
            }

            const double radToDeg = 180.0 / Math.PI;

            return new[]
            {
                roll * radToDeg,
                pitch * radToDeg,
                yaw * radToDeg
            };
        }

        private static double[,] CreateTransformMatrix(
            double[,] rotationMatrix,
            double[] translation)
        {
            double[,] transform = new double[4, 4];

            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                    transform[row, col] = rotationMatrix[row, col];

                transform[row, 3] = translation[row];
            }

            transform[3, 3] = 1.0;
            return transform;
        }
    }
}