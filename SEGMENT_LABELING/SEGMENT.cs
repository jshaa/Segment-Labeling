using OpenCvSharp;
using OpenCvSharp.Extensions;
using Seojin.RobotVision;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace SEGMENT_LABELING
{
    public partial class SEGMENT : Form
    {
        private readonly Form1 form1;
        private readonly InstanceSegmentation instanceSegmentation;
        private readonly Stopwatch stopwatch = new Stopwatch();

        // 고정 ROI가 있으면 fixedInferenceRoi를 우선 사용하고, 없으면 이전 결과 기반 추적 ROI를 사용할 수 있습니다.
        private Rect? fixedInferenceRoi;
        private Rect? adaptiveInferenceRoi;
        private bool useAdaptiveInferenceRoi;
        private double adaptiveRoiPaddingRatio = 0.60;

        private Bitmap bmp;
        private InstanceSegmentationResult segResult;

        // 3D Align 설정값
        private double[,] cameraMatrix;
        private double[] distortionCoefficients = Array.Empty<double>();
        private double objectWidthMm;
        private double objectHeightMm;
        private bool is3DAlignConfigured;

        // 포장 외곽을 3D 기준점으로 사용하는 것은 기본적으로 금지합니다.
        // Keypoint 모델 또는 강체 마커에서 얻은 네 점을 SetRigidReferenceCorners로 전달하세요.
        private bool allowSegmentationCornersFor3D;
        private Point2f[] rigidReferenceCorners = Array.Empty<Point2f>();

        public SEGMENT(
            Form1 _form1,
            InstanceSegmentation _InstanceSegmentation)
        {
            InitializeComponent();

            form1 = _form1 ?? throw new ArgumentNullException(nameof(_form1));
            instanceSegmentation = _InstanceSegmentation ??
                throw new ArgumentNullException(nameof(_InstanceSegmentation));

            // 한 프레임당 Instance Segmentation을 한 번만 수행합니다.
            instanceSegmentation.MaskThreshold = 0.40;
            instanceSegmentation.EnableTwoStageRefinement = false;
            instanceSegmentation.PreferWholeObjectCandidate = true;
            instanceSegmentation.CandidateAreaWeight = 0.35;
            instanceSegmentation.MorphologyOpenMinimumRoiSide = 80;
        }

      

        private void btn_run_Click(object sender, EventArgs e)
        {
            try
            {
                bmp = form1.currentBmp;

                if (bmp == null)
                {
                    MessageBox.Show("먼저 이미지를 불러오세요.");
                    return;
                }

                segResult?.Dispose();
                segResult = null;

                Rect? inferenceRoi = fixedInferenceRoi;

                if (!inferenceRoi.HasValue && useAdaptiveInferenceRoi)
                    inferenceRoi = adaptiveInferenceRoi;

                segResult = instanceSegmentation.Execute(bmp, inferenceRoi);

                if (segResult == null)
                {
                    if (!fixedInferenceRoi.HasValue && useAdaptiveInferenceRoi)
                        adaptiveInferenceRoi = null;

                    pictureBox1.Image = bmp;

                    MessageBox.Show("검출된 객체가 없습니다.");
                    return;
                }

                if (!fixedInferenceRoi.HasValue && useAdaptiveInferenceRoi)
                {
                    adaptiveInferenceRoi = ExpandRect(
                        segResult.BoundingBox,
                        bmp.Width,
                        bmp.Height,
                        adaptiveRoiPaddingRatio);
                }

                using (Mat colorMat = BitmapConverter.ToMat(bmp))
                {
                    DrawContourAndCorners(colorMat, segResult);

                    XYThetaAlignResult xyTheta =
                        Align3D.EstimateXYTheta(segResult.OBJECT);

                    DrawXYThetaText(colorMat, xyTheta);

                    int depthX = (int)Math.Round(xyTheta.CenterXPixel);
                    int depthY = (int)Math.Round(xyTheta.CenterYPixel);

                    float zMm = 0;

                    Console.WriteLine(
                        $"XYθ ALIGN - X:{xyTheta.CenterXPixel:F2}px, " +
                        $"Y:{xyTheta.CenterYPixel:F2}px, " +
                        $"Theta:{xyTheta.ThetaDeg:F3}°");

                    if (is3DAlignConfigured)
                    {
                        Point2f[] poseCorners = Resolve3DAlignCorners();

                    }
                    else
                    {
                        DrawStatusText(
                            colorMat,
                            "3D Align: calibration not configured",
                            50);
                    }

                    SetPictureBoxImage(BitmapConverter.ToBitmap(colorMat));
                }


                stopwatch.Stop();



                Console.WriteLine(
                    $"{stopwatch.ElapsedMilliseconds}ms, " +
                    $"Confidence:{segResult.Confidence:F4}, " +
                    $"SinglePassROI:{segResult.UsedInferenceRoi}");

                //Bitmap resultImage = new Bitmap(bmp);

                //using (Graphics g = Graphics.FromImage(resultImage))
                //using (Pen pen = new Pen(Color.Red, 3))
                //using (Pen refinementPen = new Pen(Color.Cyan, 1))
                //using (Font font = new Font("Arial", 14, FontStyle.Bold))
                //using (SolidBrush brush = new SolidBrush(Color.Yellow))
                //{
                //    g.DrawRectangle(
                //        pen,
                //        segResult.BoundingBox.X,
                //        segResult.BoundingBox.Y,
                //        segResult.BoundingBox.Width,
                //        segResult.BoundingBox.Height);

                //    if (segResult.UsedInferenceRoi &&
                //        segResult.InferenceRoi.Width > 0 &&
                //        segResult.InferenceRoi.Height > 0)
                //    {
                //        g.DrawRectangle(
                //            refinementPen,
                //            segResult.InferenceRoi.X,
                //            segResult.InferenceRoi.Y,
                //            segResult.InferenceRoi.Width,
                //            segResult.InferenceRoi.Height);
                //    }

                //    g.DrawString(
                //        $"{segResult.Confidence * 100:F1}%",
                //        font,
                //        brush,
                //        new PointF(
                //            segResult.BoundingBox.X,
                //            Math.Max(0, segResult.BoundingBox.Y - 25)));
                //}

                //SetPictureBoxImage(resultImage);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "추론 오류");
            }
        }

        /// <summary>
        /// Contour, XYθ Align 및 설정된 경우 강체 기준점 기반 3D 자세를 표시합니다.
        /// </summary>
        private void btn_contour_Click(object sender, EventArgs e)
        {
            try
            {
                if (segResult == null)
                {
                    MessageBox.Show("먼저 RUN을 실행하세요.");
                    return;
                }

                bmp = form1.currentBmp;

                if (bmp == null)
                    return;


            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Contour / Align 오류");
            }
        }

        private Point2f[] Resolve3DAlignCorners()
        {
            if (rigidReferenceCorners != null &&
                rigidReferenceCorners.Length == 4)
            {
                return InstanceSegmentation.OrderCorners(rigidReferenceCorners);
            }

            if (allowSegmentationCornersFor3D &&
                segResult != null &&
                segResult.IsPerspectiveQuadrilateral &&
                segResult.Corners != null &&
                segResult.Corners.Length == 4)
            {
                return InstanceSegmentation.OrderCorners(segResult.Corners);
            }

            return Array.Empty<Point2f>();
        }

        /// <summary>
        /// 별도 버튼을 추가할 경우 Click 이벤트에 연결하여 원근 보정 영상을 확인할 수 있습니다.
        /// </summary>
        private void btn_rectify_Click(object sender, EventArgs e)
        {
            try
            {
                if (segResult == null ||
                    segResult.Corners == null ||
                    segResult.Corners.Length != 4)
                {
                    MessageBox.Show("먼저 객체와 네 모서리를 검출하세요.");
                    return;
                }

                bmp = form1.currentBmp;

                if (bmp == null)
                    return;

                const int outputWidth = 400;
                int outputHeight;

                if (objectWidthMm > 0.0 && objectHeightMm > 0.0)
                {
                    outputHeight = Math.Max(
                        2,
                        (int)Math.Round(
                            outputWidth * objectHeightMm / objectWidthMm));
                }
                else
                {
                    outputHeight = 600;
                }

                using (Mat source = BitmapConverter.ToMat(bmp))
                using (Mat aligned = Align3D.RectifyPerspective(
                    source,
                    segResult.Corners,
                    outputWidth,
                    outputHeight))
                {
                    SetPictureBoxImage(BitmapConverter.ToBitmap(aligned));
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "원근 보정 오류");
            }
        }

        private static void DrawContourAndCorners(
            Mat image,
            InstanceSegmentationResult result)
        {
            int minImageSide = Math.Min(image.Width, image.Height);
            int minObjectSide = Math.Min(
                result.BoundingBox.Width,
                result.BoundingBox.Height);

            bool smallPreview =
                minImageSide < 300 ||
                minObjectSide < 60;

            int contourThickness = smallPreview ? 1 : 2;
            int cornerRadius = smallPreview ? 2 : 5;
            int cornerThickness = smallPreview ? 1 : 2;

            if (result.Contour != null && result.Contour.Length >= 3)
            {
                Cv2.DrawContours(
                    image,
                    new[] { result.Contour },
                    -1,
                    new Scalar(0, 0, 255),
                    contourThickness,
                    LineTypes.AntiAlias);
            }

            if (result.Corners == null || result.Corners.Length != 4)
                return;

            OpenCvSharp.Point[] cornerPoints = result.Corners
                .Select(
                    p => new OpenCvSharp.Point(
                        (int)Math.Round(p.X),
                        (int)Math.Round(p.Y)))
                .ToArray();

            Cv2.Polylines(
                image,
                new[] { cornerPoints },
                true,
                new Scalar(0, 255, 0),
                cornerThickness,
                LineTypes.AntiAlias);

            string[] names = { "TL", "TR", "BR", "BL" };

            for (int i = 0; i < cornerPoints.Length; i++)
            {
                Cv2.Circle(
                    image,
                    cornerPoints[i],
                    cornerRadius,
                    new Scalar(255, 0, 255),
                    -1,
                    LineTypes.AntiAlias);

                // 작은 Preview 또는 작은 객체에서는 문자가 모서리를 가리므로 표시하지 않습니다.
                if (smallPreview)
                    continue;

                Cv2.PutText(
                    image,
                    names[i],
                    new OpenCvSharp.Point(
                        cornerPoints[i].X + 6,
                        cornerPoints[i].Y - 6),
                    HersheyFonts.HersheySimplex,
                    0.45,
                    new Scalar(255, 0, 255),
                    1,
                    LineTypes.AntiAlias);
            }
        }

        private static void DrawXYThetaText(
            Mat image,
            XYThetaAlignResult result)
        {
            if (Math.Min(image.Width, image.Height) < 200)
                return;

            string text =
                $"XYTheta(px/deg) X:{result.CenterXPixel:F1} " +
                $"Y:{result.CenterYPixel:F1} T:{result.ThetaDeg:F2}";

            Cv2.PutText(
                image,
                text,
                new OpenCvSharp.Point(10, 25),
                HersheyFonts.HersheySimplex,
                0.50,
                new Scalar(0, 255, 255),
                2,
                LineTypes.AntiAlias);
        }

        private static void DrawPoseText(
            Mat image,
            Pose3DResult pose,
            int startY)
        {
            if (Math.Min(image.Width, image.Height) < 200)
                return;

            string line1 =
                $"T(mm) X:{pose.Xmm:F2} Y:{pose.Ymm:F2} Z:{pose.Zmm:F2}";

            string line2 =
                $"R(deg) X:{pose.RollDeg:F2} Y:{pose.PitchDeg:F2} Z:{pose.YawDeg:F2}";

            string line3 =
                $"Reprojection:{pose.ReprojectionErrorPx:F2}px  PnP:{pose.SolverUsed}";

            Cv2.PutText(
                image,
                line1,
                new OpenCvSharp.Point(10, startY),
                HersheyFonts.HersheySimplex,
                0.50,
                new Scalar(0, 255, 255),
                2,
                LineTypes.AntiAlias);

            Cv2.PutText(
                image,
                line2,
                new OpenCvSharp.Point(10, startY + 23),
                HersheyFonts.HersheySimplex,
                0.50,
                new Scalar(0, 255, 255),
                2,
                LineTypes.AntiAlias);

            Cv2.PutText(
                image,
                line3,
                new OpenCvSharp.Point(10, startY + 46),
                HersheyFonts.HersheySimplex,
                0.50,
                new Scalar(0, 255, 255),
                2,
                LineTypes.AntiAlias);
        }

        private static void DrawStatusText(
            Mat image,
            string message,
            int y)
        {
            if (Math.Min(image.Width, image.Height) < 200)
                return;

            Cv2.PutText(
                image,
                message,
                new OpenCvSharp.Point(10, y),
                HersheyFonts.HersheySimplex,
                0.50,
                new Scalar(0, 255, 255),
                2,
                LineTypes.AntiAlias);
        }

        private static Rect ExpandRect(
            Rect box,
            int imageWidth,
            int imageHeight,
            double paddingRatio)
        {
            double safeRatio = Math.Max(0.0, paddingRatio);

            int paddingX = Math.Max(8, (int)Math.Round(box.Width * safeRatio));
            int paddingY = Math.Max(8, (int)Math.Round(box.Height * safeRatio));

            int left = Math.Max(0, box.X - paddingX);
            int top = Math.Max(0, box.Y - paddingY);
            int right = Math.Min(imageWidth, box.X + box.Width + paddingX);
            int bottom = Math.Min(imageHeight, box.Y + box.Height + paddingY);

            return new Rect(
                left,
                top,
                Math.Max(0, right - left),
                Math.Max(0, bottom - top));
        }

        private void SetPictureBoxImage(Bitmap newImage)
        {
            Image previous = pictureBox1.Image;
            pictureBox1.Image = newImage;
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            previous?.Dispose();
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            segResult?.Dispose();
            segResult = null;

            Image image = pictureBox1.Image;
            pictureBox1.Image = null;
            image?.Dispose();

            base.OnFormClosed(e);
        }
    }
}
