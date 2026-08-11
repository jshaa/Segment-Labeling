namespace SEGMENT_LABELING
{
    partial class SEGMENT
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.btn_run = new System.Windows.Forms.Button();
            this.btn_contour = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.pictureBox1.Location = new System.Drawing.Point(23, 12);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(545, 425);
            this.pictureBox1.TabIndex = 3;
            this.pictureBox1.TabStop = false;
            // 
            // btn_run
            // 
            this.btn_run.Location = new System.Drawing.Point(682, 12);
            this.btn_run.Name = "btn_run";
            this.btn_run.Size = new System.Drawing.Size(94, 70);
            this.btn_run.TabIndex = 4;
            this.btn_run.Text = "run";
            this.btn_run.UseVisualStyleBackColor = true;
            this.btn_run.Click += new System.EventHandler(this.btn_run_Click);
            // 
            // btn_contour
            // 
            this.btn_contour.Location = new System.Drawing.Point(682, 121);
            this.btn_contour.Name = "btn_contour";
            this.btn_contour.Size = new System.Drawing.Size(94, 70);
            this.btn_contour.TabIndex = 5;
            this.btn_contour.Text = "find contour";
            this.btn_contour.UseVisualStyleBackColor = true;
            this.btn_contour.Click += new System.EventHandler(this.btn_contour_Click);
            // 
            // SEGMENT
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.btn_contour);
            this.Controls.Add(this.btn_run);
            this.Controls.Add(this.pictureBox1);
            this.Name = "SEGMENT";
            this.Text = "SEGMENT";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button btn_run;
        private System.Windows.Forms.Button btn_contour;
    }
}