using Seojin.RobotVision;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SEGMENT_LABELING
{
    public partial class Form1 : Form
    {

        private readonly Train trainService = new Train();
        private readonly Params trainingParams = new Params();

        private bool isTraining;

        private readonly Dictionary<string, List<List<PointF>>> imagePolygons =
            new Dictionary<string, List<List<PointF>>>(StringComparer.OrdinalIgnoreCase);

        // 빈 Polygon 목록과 아직 로드하지 않은 상태를 구분하기 위한 목록입니다.
        private readonly HashSet<string> labelLoadCompleted =
            new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        private readonly List<PointF> currentPolygon = new List<PointF>();

        private string[] filenames = Array.Empty<string>();
        public Bitmap currentBmp;
        private RectangleF imgRect;
        private RectangleF fitImgRect;
        private PointF currentMousePos;
        private bool isDrawingPolygon;
        private int index;
        private float zoomScale = 1.0f;

        private const float MinZoomScale = 0.25f;
        private const float MaxZoomScale = 20.0f;
        private const int DatasetSplitSeed = 42;

        // "LOAD PYTHON" 버튼으로 선택한 학습 스크립트 경로
        private string TrainScriptPath = string.Empty;

        // SEGMENT_PYTHON_EXE 환경 변수가 있으면 우선 사용하고,
        // 없으면 현재 프로젝트에서 사용하는 Segment 가상환경을 자동 탐색합니다.
        private readonly string PythonExePath = Path.Combine(Application.StartupPath, "Trainer.exe");

        private string DataYamlPath = string.Empty;

        public InstanceSegmentation InstanceSegmentation;

        public Form1()
        {
            InitializeComponent();

            Init_Drawing();
            Init_CosineLearning();
            Init_Batch();
            Init_Model();

            ApplyParamsToControls();

            trainService.OnLogMessage += Train_OnLogMessage;
            picturebox_main.MouseEnter += delegate { picturebox_main.Focus(); };
            picturebox_main.MouseMove += Picturebox_main_MouseMove;
            picturebox_main.Paint += Picturebox_main_Paint;
            picturebox_main.Resize += picturebox_main_Resize;
            picturebox_main.MouseWheel += Picturebox_main_MouseWheel;
            picturebox_main.MouseDoubleClick += Picturebox_main_MouseDoubleClick;
            picturebox_main.MouseClick += Picturebox_main_MouseClick;
            picturebox_main.MouseMove += Picturebox_main_MouseMove;

            typeof(PictureBox).InvokeMember(
                "DoubleBuffered",
                System.Reflection.BindingFlags.SetProperty |
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic,
                null,
                picturebox_main,
                new object[] { true });
        }


        private void Train_OnLogMessage(string message)
        {
            Debug.WriteLine(message);
        }

        private void Init_Drawing()
        {
            cmb_DrawingTool.Items.Clear();
            cmb_DrawingTool.Items.Add("Polygon");
            cmb_DrawingTool.DropDownStyle = ComboBoxStyle.DropDownList;
            cmb_DrawingTool.SelectedIndex = 0;
        }

        private void Init_CosineLearning()
        {
            cmb_CosineLearning.Items.Clear();
            cmb_CosineLearning.Items.Add("TRUE");
            cmb_CosineLearning.Items.Add("FALSE");
            cmb_CosineLearning.DropDownStyle = ComboBoxStyle.DropDownList;
        }

        private void Init_Batch()
        {
            cmb_Batch.Items.Clear();
            cmb_Batch.Items.AddRange(new object[] { "4", "8", "16", "32" });
            cmb_Batch.DropDownStyle = ComboBoxStyle.DropDownList;
        }

        private void Init_Model()
        {
            cmb_model.Items.Clear();
            cmb_model.Items.Add("yolo11m-seg.pt");  // Medium: 속도와 정확도 균형
            cmb_model.Items.Add("yolo11x-seg.pt");  // Extra Large: 최고 정확도, 높은 VRAM 사용
            cmb_model.Items.Add("yolo11s-seg.pt");  // Small: 속도 중심
            cmb_model.Items.Add("yolo11n-seg.pt");  // Nano: 가장 빠르고 가벼움
            cmb_model.Items.Add("yolo11l-seg.pt");  // Large: 정확도 중심

            cmb_model.Items.Add("yolov8m-seg.pt");  // Medium: 속도와 정확도 균형 - YOLOv8 기반
            cmb_model.Items.Add("yolov8l-seg.pt");  // Large: 정확도 중심 - YOLOv8 기반
            cmb_model.DropDownStyle = ComboBoxStyle.DropDownList;
            cmb_model.SelectedIndex = 0;
        }

        private void ApplyParamsToControls()
        {
            nud_Epoch.Value = ClampDecimal(trainingParams.Epoch, nud_Epoch.Minimum, nud_Epoch.Maximum);
            nud_patience.Value = ClampDecimal(trainingParams.Patience, nud_patience.Minimum, nud_patience.Maximum);
            nud_Degree.Value = ClampDecimal((decimal)trainingParams.Degree, nud_Degree.Minimum, nud_Degree.Maximum);
            nud_LearningRate.Value = ClampDecimal(
                (decimal)trainingParams.LearningRate,
                nud_LearningRate.Minimum,
                nud_LearningRate.Maximum);

            string batch = trainingParams.Batch.ToString(CultureInfo.InvariantCulture);
            cmb_Batch.SelectedItem = cmb_Batch.Items.Contains(batch) ? batch : "16";
            cmb_CosineLearning.SelectedItem = trainingParams.CosineLearningRate ? "TRUE" : "FALSE";
            cmb_model.SelectedItem = trainingParams.Model;
        }

        private static decimal ClampDecimal(decimal value, decimal minimum, decimal maximum)
        {
            return Math.Min(maximum, Math.Max(minimum, value));
        }

        private void btn_ImgLoad_Click(object sender, EventArgs e)
        {
            picturebox_main.Show();

            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Multiselect = true;
                ofd.Filter = "Image files|*.bmp;*.jpg;*.jpeg;*.png;*.tif;*.tiff|All files (*.*)|*.*";
                ofd.InitialDirectory = Directory.Exists(@"D:\SEGMENT") ? @"D:\SEGMENT" : Environment.CurrentDirectory;

                if (ofd.ShowDialog() != DialogResult.OK)
                    return;

                try
                {
                    string[] selectedFiles = ofd.FileNames
                        .Where(File.Exists)
                        .Distinct(StringComparer.OrdinalIgnoreCase)
                        .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                        .ToArray();

                    if (selectedFiles.Length == 0)
                    {
                        MessageBox.Show("불러올 수 있는 이미지가 없습니다.", "이미지 오류",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }

                    ResetImageSession();
                    filenames = selectedFiles;

                    foreach (string file in filenames)
                        imagePolygons[file] = new List<List<PointF>>();

                    UpdateThumbnails();

                    index = 0;
                    LoadImageAndPolygons(index);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"이미지 로드 중 오류가 발생했습니다.\n{ex.Message}", "이미지 오류",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void UpdateThumbnails()
        {
            if (flowLayoutPanel_Thumbnails == null)
                return;

            DisposeThumbnailControls();

            if (filenames == null || filenames.Length == 0)
                return;

            flowLayoutPanel_Thumbnails.SuspendLayout();

            try
            {
                for (int i = 0; i < filenames.Length; i++)
                {
                    string file = filenames[i];

                    PictureBox thumbnail = new PictureBox
                    {
                        Width = 100,
                        Height = 80,
                        SizeMode = PictureBoxSizeMode.Zoom,
                        Margin = new Padding(5),
                        Cursor = Cursors.Hand,
                        Tag = i,
                        BackColor = Color.Black
                    };

                    try
                    {
                        using (FileStream stream =
                            new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                        using (Image originalImage = Image.FromStream(stream))
                        {
                            thumbnail.Image =
                                new Bitmap(originalImage, new Size(100, 80));
                        }
                    }
                    catch (Exception ex)
                    {
                        thumbnail.Dispose();
                        Debug.WriteLine(
                            "썸네일 생성 실패: " + file + Environment.NewLine + ex.Message);
                        continue;
                    }

                    thumbnail.Click += Thumbnail_Click;
                    thumbnail.Paint += Thumbnail_Paint;
                    flowLayoutPanel_Thumbnails.Controls.Add(thumbnail);
                }
            }
            finally
            {
                flowLayoutPanel_Thumbnails.ResumeLayout();
            }
        }

        private void DisposeThumbnailControls()
        {
            if (flowLayoutPanel_Thumbnails == null)
                return;

            Control[] controls =
                flowLayoutPanel_Thumbnails.Controls.Cast<Control>().ToArray();

            flowLayoutPanel_Thumbnails.Controls.Clear();

            foreach (Control control in controls)
            {
                PictureBox thumbnail = control as PictureBox;

                if (thumbnail != null)
                {
                    thumbnail.Click -= Thumbnail_Click;
                    thumbnail.Paint -= Thumbnail_Paint;

                    Image image = thumbnail.Image;
                    thumbnail.Image = null;
                    image?.Dispose();
                }

                control.Dispose();
            }
        }

        private void Thumbnail_Paint(object sender, PaintEventArgs e)
        {
            PictureBox thumbnail = sender as PictureBox;

            if (thumbnail == null || !(thumbnail.Tag is int))
                return;

            int thumbnailIndex = (int)thumbnail.Tag;
            bool isCurrent = HasCurrentImage() && thumbnailIndex == index;

            using (Pen borderPen =
                new Pen(isCurrent ? Color.LimeGreen : Color.DimGray, isCurrent ? 4.0f : 1.0f))
            {
                borderPen.Alignment =
                    System.Drawing.Drawing2D.PenAlignment.Inset;

                Rectangle borderRectangle = new Rectangle(
                    0,
                    0,
                    Math.Max(0, thumbnail.ClientSize.Width - 1),
                    Math.Max(0, thumbnail.ClientSize.Height - 1));

                e.Graphics.DrawRectangle(borderPen, borderRectangle);
            }
        }

        private void HighlightCurrentThumbnail()
        {
            if (flowLayoutPanel_Thumbnails == null)
                return;

            Control currentThumbnail = null;

            foreach (Control control in flowLayoutPanel_Thumbnails.Controls)
            {
                control.Invalidate();

                if (control.Tag is int && (int)control.Tag == index)
                    currentThumbnail = control;
            }

            if (currentThumbnail != null)
                flowLayoutPanel_Thumbnails.ScrollControlIntoView(currentThumbnail);

            UpdateCurrentImageIndicator();
        }

        private void UpdateCurrentImageIndicator()
        {
            const string baseTitle =
                "YOLO Instance Segmentation Labeling & Training";

            if (!HasCurrentImage())
            {
                Text = baseTitle;
                return;
            }

            Text = string.Format(
                CultureInfo.InvariantCulture,
                "{0} [{1}/{2}] {3}",
                baseTitle,
                index + 1,
                filenames.Length,
                Path.GetFileName(filenames[index]));
        }

        private void ResetImageSession()
        {
            currentBmp?.Dispose();
            currentBmp = null;

            imagePolygons.Clear();
            labelLoadCompleted.Clear();
            currentPolygon.Clear();

            filenames = Array.Empty<string>();
            index = 0;
            DataYamlPath = string.Empty;
            isDrawingPolygon = false;
            currentMousePos = PointF.Empty;
            imgRect = RectangleF.Empty;
            fitImgRect = RectangleF.Empty;
            zoomScale = 1.0f;

            picturebox_main.Image = null;
            picturebox_main.Invalidate();

            DisposeThumbnailControls();
            UpdateCurrentImageIndicator();
        }

        private void btn_ImgDeleteALL_Click(object sender, EventArgs e)
        {
            ResetImageSession();
        }

        private void btn_ImgDeleteSELECT_Click(object sender, EventArgs e)
        {
            if (!HasCurrentImage())
                return;

            string removedFile = filenames[index];
            imagePolygons.Remove(removedFile);
            labelLoadCompleted.Remove(removedFile);

            List<string> list = filenames.ToList();
            list.RemoveAt(index);
            filenames = list.ToArray();
            DataYamlPath = string.Empty; // 이미지 목록이 변경되었으므로 YAML을 다시 생성해야 합니다.

            currentPolygon.Clear();
            isDrawingPolygon = false;

            if (filenames.Length == 0)
            {
                ResetImageSession();
                return;
            }

            index = Math.Min(index, filenames.Length - 1);

            UpdateThumbnails();
            LoadImageAndPolygons(index);
        }

        private void btn_Previous_Click(object sender, EventArgs e)
        {
            if (!HasCurrentImage() || index <= 0)
                return;

            CommitCurrentPolygonIfValid();
            index--;
            LoadImageAndPolygons(index);
        }

        private void btn_NEXT_Click(object sender, EventArgs e)
        {
            if (!HasCurrentImage() || index >= filenames.Length - 1)
                return;

            CommitCurrentPolygonIfValid();
            index++;
            LoadImageAndPolygons(index);
        }

        private bool HasCurrentImage()
        {
            return filenames != null && filenames.Length > 0 && index >= 0 && index < filenames.Length;
        }

        private void btn_LabelApply_Click(object sender, EventArgs e)
        {
            if (!HasCurrentImage())
                return;

            try
            {
                CommitCurrentPolygonIfValid();
                SaveAllLabels();

                MessageBox.Show("라벨 데이터 저장이 완료되었습니다.", "저장 성공",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"라벨 저장 중 오류가 발생했습니다.\n{ex.Message}", "라벨 오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void SaveAllLabels()
        {
            EnsureAllLabelsLoaded();

            foreach (string imagePath in filenames)
            {
                List<List<PointF>> polygons;

                if (!imagePolygons.TryGetValue(imagePath, out polygons))
                    polygons = new List<List<PointF>>();

                SaveLabelFile(imagePath, polygons);
                labelLoadCompleted.Add(imagePath);
            }
        }

        private void EnsureAllLabelsLoaded()
        {
            foreach (string imagePath in filenames)
            {
                if (labelLoadCompleted.Contains(imagePath))
                    continue;

                string loadError;

                if (!TryLoadLabelFile(imagePath, out loadError))
                {
                    throw new InvalidDataException(
                        "기존 라벨 파일을 읽지 못해 저장을 중단했습니다.\n" +
                        loadError);
                }

                labelLoadCompleted.Add(imagePath);
            }
        }

        private static void SaveLabelFile(string imagePath, IEnumerable<List<PointF>> polygons)
        {
            int imageWidth;
            int imageHeight;

            using (Image image = Image.FromFile(imagePath))
            {
                imageWidth = image.Width;
                imageHeight = image.Height;
            }

            List<string> lines = new List<string>();

            foreach (List<PointF> sourcePolygon in polygons)
            {
                List<PointF> polygon = CleanPolygon(sourcePolygon, imageWidth, imageHeight);
                if (polygon.Count < 3)
                    continue;

                StringBuilder line = new StringBuilder("0");
                foreach (PointF point in polygon)
                {
                    float normalizedX = Math.Max(0.0f, Math.Min(1.0f, point.X / imageWidth));
                    float normalizedY = Math.Max(0.0f, Math.Min(1.0f, point.Y / imageHeight));

                    line.Append(' ');
                    line.Append(normalizedX.ToString("F6", CultureInfo.InvariantCulture));
                    line.Append(' ');
                    line.Append(normalizedY.ToString("F6", CultureInfo.InvariantCulture));
                }

                lines.Add(line.ToString());
            }

            string txtPath = Path.ChangeExtension(imagePath, ".txt");
            string tempPath = txtPath + ".tmp";

            File.WriteAllLines(tempPath, lines, new UTF8Encoding(false));

            if (File.Exists(txtPath))
                File.Delete(txtPath);

            File.Move(tempPath, txtPath);
        }

        private static List<PointF> CleanPolygon(IEnumerable<PointF> points, int width, int height)
        {
            List<PointF> cleaned = new List<PointF>();

            foreach (PointF point in points)
            {
                PointF clamped = new PointF(
                    Math.Max(0.0f, Math.Min(width - 1.0f, point.X)),
                    Math.Max(0.0f, Math.Min(height - 1.0f, point.Y)));

                if (cleaned.Count == 0 || DistanceSquared(cleaned[cleaned.Count - 1], clamped) > 4.0f)
                    cleaned.Add(clamped);
            }

            if (cleaned.Count > 1 && DistanceSquared(cleaned[0], cleaned[cleaned.Count - 1]) <= 4.0f)
                cleaned.RemoveAt(cleaned.Count - 1);

            return cleaned;
        }

        private static float DistanceSquared(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return dx * dx + dy * dy;
        }

        private void btn_LabelDelete_Click(object sender, EventArgs e)
        {
            if (!HasCurrentImage())
                return;

            string currentFile = filenames[index];
            imagePolygons[currentFile].Clear();
            labelLoadCompleted.Add(currentFile);

            currentPolygon.Clear();
            isDrawingPolygon = false;

            try
            {
                // 빈 라벨 파일을 저장해 기존 라벨이 다시 복원되는 문제를 방지합니다.
                SaveLabelFile(currentFile, imagePolygons[currentFile]);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"라벨 삭제 내용을 저장하지 못했습니다.\n{ex.Message}", "라벨 오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            picturebox_main.Invalidate();
        }

        private void LoadImageAndPolygons(int imgIndex)
        {
            if (filenames == null || filenames.Length == 0 || imgIndex < 0 || imgIndex >= filenames.Length)
                return;

            currentBmp?.Dispose();
            currentBmp = LoadBitmapUnlocked(filenames[imgIndex]);

            picturebox_main.Image = null;
            FitImageToScreen();

            string currentFile = filenames[imgIndex];
            if (!imagePolygons.ContainsKey(currentFile))
                imagePolygons[currentFile] = new List<List<PointF>>();

            if (!labelLoadCompleted.Contains(currentFile))
            {
                string loadError;

                if (TryLoadLabelFile(currentFile, out loadError))
                {
                    labelLoadCompleted.Add(currentFile);
                }
                else
                {
                    MessageBox.Show(
                        loadError,
                        "라벨 오류",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                }
            }

            currentPolygon.Clear();
            isDrawingPolygon = false;
            currentMousePos = PointF.Empty;
            picturebox_main.Image = currentBmp;
            picturebox_main.Invalidate();

            HighlightCurrentThumbnail();
        }

        private static Bitmap LoadBitmapUnlocked(string path)
        {
            using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (Image image = Image.FromStream(stream))
            {
                return new Bitmap(image);
            }
        }

        private bool TryLoadLabelFile(string imagePath, out string error)
        {
            error = string.Empty;
            string txtPath = Path.ChangeExtension(imagePath, ".txt");

            if (!File.Exists(txtPath))
            {
                imagePolygons[imagePath] = new List<List<PointF>>();
                return true;
            }

            try
            {
                int imageWidth;
                int imageHeight;

                if (currentBmp != null &&
                    HasCurrentImage() &&
                    string.Equals(
                        imagePath,
                        filenames[index],
                        StringComparison.OrdinalIgnoreCase))
                {
                    imageWidth = currentBmp.Width;
                    imageHeight = currentBmp.Height;
                }
                else
                {
                    using (Image image = Image.FromFile(imagePath))
                    {
                        imageWidth = image.Width;
                        imageHeight = image.Height;
                    }
                }

                List<List<PointF>> loadedPolygons =
                    new List<List<PointF>>();

                int lineNumber = 0;

                foreach (string line in File.ReadLines(txtPath))
                {
                    lineNumber++;

                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    string[] tokens = line.Split(
                        new[] { ' ', '\t' },
                        StringSplitOptions.RemoveEmptyEntries);

                    if (tokens.Length < 7 || tokens.Length % 2 == 0)
                    {
                        error = string.Format(
                            CultureInfo.InvariantCulture,
                            "{0} {1}행: Polygon 형식이 올바르지 않습니다.",
                            Path.GetFileName(txtPath),
                            lineNumber);
                        return false;
                    }

                    int classId;

                    if (!int.TryParse(
                            tokens[0],
                            NumberStyles.Integer,
                            CultureInfo.InvariantCulture,
                            out classId) ||
                        classId != 0)
                    {
                        error = string.Format(
                            CultureInfo.InvariantCulture,
                            "{0} {1}행: 클래스 ID는 0이어야 합니다.",
                            Path.GetFileName(txtPath),
                            lineNumber);
                        return false;
                    }

                    List<PointF> polygon = new List<PointF>();

                    for (int i = 1; i + 1 < tokens.Length; i += 2)
                    {
                        float normalizedX;
                        float normalizedY;

                        if (!float.TryParse(
                                tokens[i],
                                NumberStyles.Float,
                                CultureInfo.InvariantCulture,
                                out normalizedX) ||
                            !float.TryParse(
                                tokens[i + 1],
                                NumberStyles.Float,
                                CultureInfo.InvariantCulture,
                                out normalizedY))
                        {
                            error = string.Format(
                                CultureInfo.InvariantCulture,
                                "{0} {1}행: 숫자 형식이 올바르지 않습니다.",
                                Path.GetFileName(txtPath),
                                lineNumber);
                            return false;
                        }

                        if (normalizedX < 0.0f || normalizedX > 1.0f ||
                            normalizedY < 0.0f || normalizedY > 1.0f)
                        {
                            error = string.Format(
                                CultureInfo.InvariantCulture,
                                "{0} {1}행: 좌표는 0~1 범위여야 합니다.",
                                Path.GetFileName(txtPath),
                                lineNumber);
                            return false;
                        }

                        polygon.Add(
                            new PointF(
                                normalizedX * imageWidth,
                                normalizedY * imageHeight));
                    }

                    if (polygon.Count < 3)
                    {
                        error = string.Format(
                            CultureInfo.InvariantCulture,
                            "{0} {1}행: Polygon 점은 최소 3개여야 합니다.",
                            Path.GetFileName(txtPath),
                            lineNumber);
                        return false;
                    }

                    loadedPolygons.Add(polygon);
                }

                imagePolygons[imagePath] = loadedPolygons;
                return true;
            }
            catch (Exception ex)
            {
                error =
                    "라벨 파일 로드 실패: " +
                    Path.GetFileName(txtPath) +
                    Environment.NewLine +
                    ex.Message;
                return false;
            }
        }

        private void FitImageToScreen()
        {
            if (currentBmp == null || picturebox_main.ClientSize.Width <= 0 || picturebox_main.ClientSize.Height <= 0)
                return;

            float pbW = picturebox_main.ClientSize.Width;
            float pbH = picturebox_main.ClientSize.Height;
            float ratio = Math.Min(pbW / currentBmp.Width, pbH / currentBmp.Height);

            float drawW = currentBmp.Width * ratio;
            float drawH = currentBmp.Height * ratio;
            float offsetX = (pbW - drawW) / 2.0f;
            float offsetY = (pbH - drawH) / 2.0f;

            fitImgRect = new RectangleF(offsetX, offsetY, drawW, drawH);
            imgRect = fitImgRect;
            zoomScale = 1.0f;
        }

        private void Picturebox_main_MouseWheel(object sender, MouseEventArgs e)
        {
            if (currentBmp == null || imgRect.Width <= 0.0f || imgRect.Height <= 0.0f)
                return;

            float requestedScale = e.Delta > 0 ? zoomScale * 1.2f : zoomScale / 1.2f;
            float targetScale = Math.Max(MinZoomScale, Math.Min(MaxZoomScale, requestedScale));
            float scaleFactor = targetScale / zoomScale;

            if (Math.Abs(scaleFactor - 1.0f) < 0.0001f)
                return;

            float newW = imgRect.Width * scaleFactor;
            float newH = imgRect.Height * scaleFactor;
            float newX = e.X - ((e.X - imgRect.X) * scaleFactor);
            float newY = e.Y - ((e.Y - imgRect.Y) * scaleFactor);

            imgRect = new RectangleF(newX, newY, newW, newH);
            zoomScale = targetScale;
            picturebox_main.Invalidate();
        }

        private void Picturebox_main_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            // 더블클릭 직전에 MouseClick으로 추가된 마지막 점을 제거합니다.
            if (isDrawingPolygon && currentPolygon.Count > 0)
                currentPolygon.RemoveAt(currentPolygon.Count - 1);

            FitImageToScreen();
            picturebox_main.Invalidate();
        }

        private void Picturebox_main_MouseClick(object sender, MouseEventArgs e)
        {
            if (currentBmp == null || !HasCurrentImage())
                return;

            if (!string.Equals(cmb_DrawingTool.SelectedItem?.ToString(), "Polygon", StringComparison.Ordinal))
                return;

            PointF imagePoint;
            if (e.Button == MouseButtons.Left && TryScreenToImage(e.Location, out imagePoint))
            {
                isDrawingPolygon = true;
                currentPolygon.Add(imagePoint);
                currentMousePos = imagePoint;
                picturebox_main.Invalidate();
            }
            else if (e.Button == MouseButtons.Right)
            {
                CommitCurrentPolygonIfValid();
                picturebox_main.Invalidate();
            }
        }

        private void CommitCurrentPolygonIfValid()
        {
            if (!HasCurrentImage())
                return;

            if (currentPolygon.Count >= 3)
            {
                string currentFile = filenames[index];
                imagePolygons[currentFile].Add(new List<PointF>(currentPolygon));
                labelLoadCompleted.Add(currentFile);
            }

            currentPolygon.Clear();
            isDrawingPolygon = false;
        }

        private bool TryScreenToImage(Point screenPoint, out PointF imagePoint)
        {
            imagePoint = PointF.Empty;

            if (currentBmp == null || imgRect.Width <= 0.0f || imgRect.Height <= 0.0f ||
                !imgRect.Contains(screenPoint))
            {
                return false;
            }

            float x = (screenPoint.X - imgRect.X) * currentBmp.Width / imgRect.Width;
            float y = (screenPoint.Y - imgRect.Y) * currentBmp.Height / imgRect.Height;

            imagePoint = new PointF(
                Math.Max(0.0f, Math.Min(currentBmp.Width - 1.0f, x)),
                Math.Max(0.0f, Math.Min(currentBmp.Height - 1.0f, y)));
            return true;
        }

        private void Picturebox_main_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isDrawingPolygon)
                return;

            PointF point;
            if (TryScreenToImage(e.Location, out point))
            {
                currentMousePos = point;
                picturebox_main.Invalidate();
            }
        }

        private void Picturebox_main_Paint(object sender, PaintEventArgs e)
        {
            if (currentBmp == null || !HasCurrentImage())
                return;

            e.Graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            e.Graphics.DrawImage(currentBmp, imgRect);
            e.Graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            Func<List<PointF>, PointF[]> convertToScreenPoints = originalPoints =>
                originalPoints.Select(point => new PointF(
                    point.X * imgRect.Width / currentBmp.Width + imgRect.X,
                    point.Y * imgRect.Height / currentBmp.Height + imgRect.Y)).ToArray();

            string currentFile = filenames[index];
            List<List<PointF>> polygons;
            if (imagePolygons.TryGetValue(currentFile, out polygons))
            {
                using (SolidBrush fillBrush = new SolidBrush(Color.FromArgb(100, 0, 255, 0)))
                using (Pen edgePen = new Pen(Color.LimeGreen, 2.0f))
                {
                    foreach (List<PointF> polygon in polygons)
                    {
                        if (polygon.Count < 3)
                            continue;

                        PointF[] screenPoints = convertToScreenPoints(polygon);
                        e.Graphics.FillPolygon(fillBrush, screenPoints);
                        e.Graphics.DrawPolygon(edgePen, screenPoints);
                    }
                }
            }

            if (isDrawingPolygon && currentPolygon.Count > 0)
            {
                using (Pen drawingPen = new Pen(Color.Red, 2.0f))
                {
                    PointF[] currentScreenPoints = convertToScreenPoints(currentPolygon);
                    if (currentScreenPoints.Length > 1)
                        e.Graphics.DrawLines(drawingPen, currentScreenPoints);

                    PointF screenMousePos = new PointF(
                        currentMousePos.X * imgRect.Width / currentBmp.Width + imgRect.X,
                        currentMousePos.Y * imgRect.Height / currentBmp.Height + imgRect.Y);

                    e.Graphics.DrawLine(drawingPen, currentScreenPoints[currentScreenPoints.Length - 1], screenMousePos);
                }
            }
        }

        private void picturebox_main_Resize(object sender, EventArgs e)
        {
            if (currentBmp == null)
                return;

            FitImageToScreen();
            picturebox_main.Invalidate();
        }

        private void btn_LoadPython_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Python files (*.py)|*.py|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                    TrainScriptPath = ofd.FileName;
            }
        }

        private async void btn_Train_Click(object sender, EventArgs e)
        {
            if (isTraining || trainService.IsRunning)
                return;

            if (!ValidateTrainingFiles(requireDataset: true))
                return;

            using (FolderBrowserDialog folderDialog = new FolderBrowserDialog())
            {
                folderDialog.Description =
                    "학습 결과(가중치 및 ONNX 모델)를 저장할 폴더를 선택하세요.";

                if (HasCurrentImage())
                    folderDialog.SelectedPath =
                        Path.GetDirectoryName(filenames[0]);

                if (folderDialog.ShowDialog() != DialogResult.OK)
                    return;

                SyncParamsFromControls();
                SetExecutionState(true, "TRAINING...");

                try
                {
                    TrainResult result = await trainService.TrainAsync(
                        PythonExePath,
                        folderDialog.SelectedPath,
                        DataYamlPath,
                        trainingParams);

                    if (result.AutoExportFailed)
                    {
                        MessageBox.Show(
                            "모델 학습은 완료되었지만 ONNX 자동 변환에 실패했습니다.\n" +
                            "best.pt: " + result.BestPtPath,
                            "학습 완료 / 변환 경고",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Warning);
                    }
                    else
                    {
                        StringBuilder message = new StringBuilder();
                        message.AppendLine("모델 학습이 완료되었습니다.");

                        if (!string.IsNullOrWhiteSpace(result.BestPtPath))
                            message.AppendLine("best.pt: " + result.BestPtPath);

                        if (!string.IsNullOrWhiteSpace(result.OnnxPath))
                            message.AppendLine("ONNX: " + result.OnnxPath);

                        MessageBox.Show(
                            message.ToString(),
                            "학습 완료",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Information);

                        InstanceSegmentation = new InstanceSegmentation();
                        InstanceSegmentation.Initialize(result.OnnxPath);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show(
                        "학습 실행 오류가 발생했습니다.\n" + ex.Message,
                        "학습 오류",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);
                }
                finally
                {
                    SetExecutionState(false, "TRAIN");
                }
            }
        }

        private void btn_ONNXExport_Click(object sender, EventArgs e)
        {
            if (isTraining || trainService.IsRunning)
                return;

            if (!ValidateTrainingFiles(requireDataset: false))
                return;

            InstanceSegmentation = new InstanceSegmentation();
            InstanceSegmentation.Initialize(@"D:\SEGMENT\Run_20260726_230120\weights\best.onnx");
            Console.WriteLine("모델 로드 완료");

            //using (OpenFileDialog fileDialog = new OpenFileDialog())
            //{
            //    fileDialog.Filter =
            //        "PyTorch weights (*.pt)|*.pt|All files (*.*)|*.*";
            //    fileDialog.Title = "변환할 best.pt 파일을 선택하세요";

            //    if(fileDialog.ShowDialog() == DialogResult.OK)
            //    {

            //    }

            //if (fileDialog.ShowDialog() != DialogResult.OK)
            //    return;

            //SyncParamsFromControls();
            //SetExecutionState(true, "EXPORTING...");

            //try
            //{
            //    TrainResult result = await trainService.ExportOnnxAsync(PythonExePath, fileDialog.FileName, trainingParams);

            //    MessageBox.Show(
            //        string.IsNullOrWhiteSpace(result.OnnxPath)
            //            ? "ONNX 변환이 완료되었습니다."
            //            : "ONNX 변환이 완료되었습니다.\n" + result.OnnxPath,
            //        "ONNX 변환 완료",
            //        MessageBoxButtons.OK,
            //        MessageBoxIcon.Information);
            //}
            //catch (Exception ex)
            //{
            //    MessageBox.Show(
            //        "ONNX 변환 오류가 발생했습니다.\n" + ex.Message,
            //        "ONNX 변환 오류",
            //        MessageBoxButtons.OK,
            //        MessageBoxIcon.Error);
            //}
            //finally
            //{
            //    SetExecutionState(false, "TRAIN");
            //}
            //}
        }

        private bool ValidateTrainingFiles(bool requireDataset)
        {
            bool pythonIsPathCommand =
        PythonExePath.IndexOf(Path.DirectorySeparatorChar) < 0 &&
        PythonExePath.IndexOf(Path.AltDirectorySeparatorChar) < 0;

            if (!pythonIsPathCommand && !File.Exists(PythonExePath))
            {
                MessageBox.Show(
                    "패키징된 실행 파일을 찾을 수 없습니다.\n" + PythonExePath + "\n\n" +
                    "SegmentTrain_test.exe와 _internal 폴더를 C# 실행 폴더에 넣어주세요.",
                    "실행 파일 누락",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return false;
            }

            if (requireDataset &&
                (string.IsNullOrWhiteSpace(DataYamlPath) ||
                 !File.Exists(DataYamlPath)))
            {
                MessageBox.Show(
                    "먼저 [LABELS to YAML] 버튼으로 data.yaml을 생성하세요.",
                    "데이터셋 미설정",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return false;
            }

            return true;
        }

        private void SyncParamsFromControls()
        {
            int batch;

            if (!int.TryParse(
                    cmb_Batch.SelectedItem?.ToString(),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out batch))
            {
                batch = 16;
            }

            trainingParams.Model =
                cmb_model.SelectedItem?.ToString() ?? "yolo11m-seg.pt";
            trainingParams.Epoch = (int)nud_Epoch.Value;
            trainingParams.Batch = batch;
            trainingParams.Patience = (int)nud_patience.Value;
            trainingParams.Degree = (float)nud_Degree.Value;
            trainingParams.LearningRate = (float)nud_LearningRate.Value;
            trainingParams.CosineLearningRate = string.Equals(
                cmb_CosineLearning.SelectedItem?.ToString(),
                "TRUE",
                StringComparison.OrdinalIgnoreCase);
        }

        private void Cmb_Drawing_ValueChanged(object sender, EventArgs e)
        {
            // 현재는 Polygon 도구만 지원합니다.
        }

        private void Cmb_Batch_ValueChanged(object sender, EventArgs e)
        {
            int batch;
            if (int.TryParse(cmb_Batch.SelectedItem?.ToString(), out batch))
                trainingParams.Batch = batch;
        }

        private void Cmb_Cosine_ValueChanged(object sender, EventArgs e)
        {
            trainingParams.CosineLearningRate = string.Equals(
                cmb_CosineLearning.SelectedItem?.ToString(),
                "TRUE",
                StringComparison.OrdinalIgnoreCase);
        }

        private void nud_Epoch_ValueChanged(object sender, EventArgs e)
        {
            trainingParams.Epoch = (int)nud_Epoch.Value;
        }

        private void nud_Patience_ValueChanged(object sender, EventArgs e)
        {
            trainingParams.Patience = (int)nud_patience.Value;
        }

        private void nud_Degree_ValueChanged(object sender, EventArgs e)
        {
            trainingParams.Degree = (float)nud_Degree.Value;
        }

        private void nud_LearningRate_ValueChanged(object sender, EventArgs e)
        {
            trainingParams.LearningRate = (float)nud_LearningRate.Value;
        }

        private void btn_YAML_Click(object sender, EventArgs e)
        {
            if (!HasCurrentImage())
            {
                MessageBox.Show("먼저 이미지를 불러오세요.", "데이터셋 미설정",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                CommitCurrentPolygonIfValid();
                SaveAllLabels();

                string datasetFolder = Path.GetDirectoryName(filenames[0]);
                CreateDataYaml(datasetFolder);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"YAML 생성 중 오류가 발생했습니다.\n{ex.Message}", "YAML 오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void CreateDataYaml(string datasetFolderPath)
        {
            if (!HasCurrentImage())
                throw new InvalidOperationException("이미지가 선택되지 않았습니다.");

            if (filenames.Length < 2)
                throw new InvalidOperationException("Train/Validation 분리를 위해 최소 2장의 이미지가 필요합니다.");

            foreach (string imagePath in filenames)
            {
                string validationError;
                if (!ValidateLabelFile(imagePath, out validationError))
                    throw new InvalidDataException(validationError);
            }

            List<string> shuffledFiles = filenames.ToList();
            Shuffle(shuffledFiles, DatasetSplitSeed);

            int validationCount = Math.Max(1, (int)Math.Round(shuffledFiles.Count * 0.2));
            int trainCount = shuffledFiles.Count - validationCount;

            List<string> trainFiles = shuffledFiles.Take(trainCount).ToList();
            List<string> valFiles = shuffledFiles.Skip(trainCount).ToList();

            string trainTxtPath = Path.Combine(datasetFolderPath, "train_list.txt");
            string valTxtPath = Path.Combine(datasetFolderPath, "val_list.txt");
            UTF8Encoding utf8WithoutBom = new UTF8Encoding(false);

            File.WriteAllLines(trainTxtPath, trainFiles.Select(ToYoloPath), utf8WithoutBom);
            File.WriteAllLines(valTxtPath, valFiles.Select(ToYoloPath), utf8WithoutBom);

            string yamlPath = EscapeYamlSingleQuoted(ToYoloPath(datasetFolderPath));
            StringBuilder yaml = new StringBuilder();
            yaml.AppendLine("path: '" + yamlPath + "'");
            yaml.AppendLine("train: train_list.txt");
            yaml.AppendLine("val: val_list.txt");
            yaml.AppendLine();
            yaml.AppendLine("nc: 1");
            yaml.AppendLine("names: ['Target']");

            string yamlFilePath = Path.Combine(datasetFolderPath, "data.yaml");
            File.WriteAllText(yamlFilePath, yaml.ToString(), utf8WithoutBom);
            DataYamlPath = yamlFilePath;

            MessageBox.Show(
                $"data.yaml이 생성되었습니다.\nTrain: {trainFiles.Count}장\nValidation: {valFiles.Count}장",
                "YAML 생성 완료",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information);
        }

        private static bool ValidateLabelFile(string imagePath, out string error)
        {
            error = string.Empty;
            string labelPath = Path.ChangeExtension(imagePath, ".txt");

            // 라벨 파일이 없는 이미지는 정상(Negative) 이미지로 허용합니다.
            if (!File.Exists(labelPath))
                return true;

            int lineNumber = 0;
            foreach (string line in File.ReadLines(labelPath))
            {
                lineNumber++;
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                string[] tokens = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length < 7 || tokens.Length % 2 == 0)
                {
                    error = $"{Path.GetFileName(labelPath)} {lineNumber}행: Polygon 형식이 올바르지 않습니다.";
                    return false;
                }

                int classId;
                if (!int.TryParse(tokens[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out classId) || classId != 0)
                {
                    error = $"{Path.GetFileName(labelPath)} {lineNumber}행: 클래스 ID는 0이어야 합니다.";
                    return false;
                }

                for (int i = 1; i < tokens.Length; i++)
                {
                    float coordinate;
                    if (!float.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out coordinate) ||
                        coordinate < 0.0f || coordinate > 1.0f)
                    {
                        error = $"{Path.GetFileName(labelPath)} {lineNumber}행: 좌표는 0~1 범위여야 합니다.";
                        return false;
                    }
                }
            }

            return true;
        }

        private static void Shuffle<T>(IList<T> list, int seed)
        {
            Random random = new Random(seed);
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                T temp = list[i];
                list[i] = list[j];
                list[j] = temp;
            }
        }

        private static string ToYoloPath(string path)
        {
            return path.Replace('\\', '/');
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            // Ctrl + Z : 라벨링 이전 상태로 되돌리기 (마지막 점 취소)
            if (keyData == (Keys.Control | Keys.Z))
            {
                // 현재 폴리곤을 그리고 있는 상태이며, 찍어둔 점이 1개 이상일 때만 동작
                if (isDrawingPolygon && currentPolygon.Count > 0)
                {
                    currentPolygon.RemoveAt(currentPolygon.Count - 1);

                    // 점을 지웠는데 남은 점이 없다면 그리기 상태 해제
                    if (currentPolygon.Count == 0)
                    {
                        isDrawingPolygon = false;
                    }

                    picturebox_main.Invalidate();
                    return true; // 키 입력 처리 완료
                }
            }
            // ESC : 라벨링 초기화 상태로 바꾸기 (현재 그리기 완전 취소)
            else if (keyData == Keys.Escape)
            {
                // 현재 폴리곤을 그리고 있는 상태일 때만 동작
                if (isDrawingPolygon)
                {
                    currentPolygon.Clear();
                    isDrawingPolygon = false;

                    picturebox_main.Invalidate();
                    return true; // 키 입력 처리 완료
                }
            }

            // 지정한 단축키가 아니라면 기본 키 처리 로직 수행
            return base.ProcessCmdKey(ref msg, keyData);
        }

        private static string EscapeYamlSingleQuoted(string value)
        {
            return value.Replace("'", "''");
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            trainService.OnLogMessage -= Train_OnLogMessage;

            currentBmp?.Dispose();
            currentBmp = null;

            DisposeThumbnailControls();
            base.OnFormClosed(e);
        }

        private void Thumbnail_Click(object sender, EventArgs e)
        {
            PictureBox thumbnail = sender as PictureBox;

            if (thumbnail == null || !(thumbnail.Tag is int))
                return;

            int clickedIndex = (int)thumbnail.Tag;

            if (clickedIndex < 0 || clickedIndex >= filenames.Length)
                return;

            CommitCurrentPolygonIfValid();
            index = clickedIndex;
            LoadImageAndPolygons(index);
        }

        private void SetExecutionState(bool running, string operationText)
        {
            isTraining = running;
            btn_Train.Enabled = !running;
            btn_ONNXExport.Enabled = !running;

            btn_Train.Text =
                running && string.Equals(
                    operationText,
                    "TRAINING...",
                    StringComparison.Ordinal)
                    ? operationText
                    : "TRAIN";

            btn_ONNXExport.Text =
                running && string.Equals(
                    operationText,
                    "EXPORTING...",
                    StringComparison.Ordinal)
                    ? operationText
                    : "ONNX EXPORT";
        }


        private void btn_test_Click(object sender, EventArgs e)
        {
            if (InstanceSegmentation == null)
            {
                MessageBox.Show(
                    "먼저 모델을 로드하세요.\n" +
                    "[TRAIN] 또는 [ONNX EXPORT] 버튼으로 모델을 초기화하세요.",
                    "모델 미초기화",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            SEGMENT sEGMENT = new SEGMENT(this, InstanceSegmentation);
            sEGMENT.Show();
        }


        private void btn_ImgSave_Click(object sender, EventArgs e)
        {
            // 1. 메모리에 캡처된 이미지가 있는지 확인
            if (currentBmp == null)
            {
                MessageBox.Show("저장할 이미지가 없습니다. 먼저 이미지를 불러오세요.",
                                "이미지 저장 오류",
                                MessageBoxButtons.OK,
                                MessageBoxIcon.Warning);
                return;
            }

            // 2. 파일 저장 다이얼로그 설정
            using (SaveFileDialog sfd = new SaveFileDialog())
            {
                sfd.Title = "이미지 저장";
                // 기존 코드의 InitialDirectory 정책과 동일하게 맞춤
                sfd.InitialDirectory = Directory.Exists(@"D:\SEGMENT") ? @"D:\SEGMENT" : Environment.CurrentDirectory;

                // 확장자 필터 (PNG를 기본으로 추천 - 비전 라벨링 시 손실 압축 방지)
                sfd.Filter = "PNG Image|*.png|BMP Image|*.bmp|JPEG Image|*.jpg";

                // 기본 파일명을 현재 시간 기반으로 설정 (예: 20260722_141530.png)
                sfd.FileName = "IMG_" + DateTime.Now.ToString("yyyyMMdd_HHmmss");

                if (sfd.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        System.Drawing.Imaging.ImageFormat format = System.Drawing.Imaging.ImageFormat.Png;
                        string ext = Path.GetExtension(sfd.FileName).ToLower();

                        switch (ext)
                        {
                            case ".bmp":
                                format = System.Drawing.Imaging.ImageFormat.Bmp;
                                break;
                            case ".jpg":
                            case ".jpeg":
                                format = System.Drawing.Imaging.ImageFormat.Jpeg;
                                break;
                            case ".png":
                            default:
                                format = System.Drawing.Imaging.ImageFormat.Png;
                                break;
                        }

                        currentBmp.Save(sfd.FileName, format);

                        // (선택사항) 저장 후 썸네일 리스트에도 즉시 반영하고 싶다면 아래 주석을 해제하고 다듬어 사용하세요.
                        /*
                        List<string> currentFiles = filenames.ToList();
                        currentFiles.Add(sfd.FileName);
                        filenames = currentFiles.ToArray();
                        imagePolygons[sfd.FileName] = new List<List<PointF>>();
                        UpdateThumbnails();
                        */

                        //MessageBox.Show("이미지 저장이 완료되었습니다.\n경로: " + sfd.FileName,
                        //                "저장 성공",
                        //                MessageBoxButtons.OK,
                        //                MessageBoxIcon.Information);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"이미지 저장 중 오류가 발생했습니다.\n{ex.Message}",
                                        "저장 실패",
                                        MessageBoxButtons.OK,
                                        MessageBoxIcon.Error);
                    }
                }
            }
        }
    }
}