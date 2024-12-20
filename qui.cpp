#include "qui.hpp"
#include <iostream>
using namespace std;
using namespace cv;
bool isVideo = false;
const std::string provinceOnnx = "./plate.onnx";
const std::string numAlphaOnnx = "./numalpha.onnx";
QString regions[] = {
    "川", "鄂", "赣", "甘", "贵", "桂",
    "黑", "沪", "冀", "津", "京", "吉",
    "辽", "鲁", "蒙", "闽", "宁", "青",
    "琼", "陕", "苏", "晋", "皖", "湘",
    "新", "豫", "渝", "粤", "云", "藏", "浙"};
QString numalpha[] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"};

Size bigSize(320, 240);
Size smallSize(440 / 7, 140);
Size plateSize(220, 70);
Qui::Qui(QWidget *parent) : QMainWindow(parent), trtProvince(provinceOnnx), trtNumAlpha(numAlphaOnnx)
{
    creatLayout();

    buildconnections();
}
void Qui::creatLayout()
{
    centerWid = new QWidget;
    mainLayout = new QVBoxLayout;
    btnLayout = new QHBoxLayout;
    imgLayout1 = new QGridLayout;
    imgLayout2 = new QHBoxLayout;
    imgLayout3 = new QVBoxLayout;
    imgLayout4 = new QVBoxLayout;
    imgLayout5 = new QHBoxLayout;

    openBtn = new QPushButton("打开图片");
    saveBtn = new QPushButton("打开摄像头");
    transformBtn = new QPushButton("提取车牌");
    detectBtn = new QPushButton("Exit");

    raw = new QLabel;
    boxed = new QLabel;
    plate = new QLabel;
    res = new QLabel;
    afterMask = new QLabel;
    hEdge = new QLabel;
    vEdge = new QLabel;
    contuor = new QLabel;
    afterTransform = new QLabel;
    canny = new QLabel;
    res->setText("识别结果");
    res->setStyleSheet("QLabel{background-color:white;color:red;font-size:25px;}");
    res->setFixedSize(220, 70);
    for (int i = 0; i < 8; i++)
        segPic[i] = new QLabel;

    btnLayout->addWidget(openBtn);
    btnLayout->addWidget(transformBtn);
    btnLayout->addWidget(saveBtn);
    btnLayout->addWidget(detectBtn);

    Mat tmpRaw = Mat::zeros(240, 320, CV_8UC3);
    Mat tmpPlate = Mat::zeros(70, 220, CV_8UC3);
    mat2LabelImg(tmpRaw, raw);
    mat2LabelImg(tmpRaw, afterMask);
    mat2LabelImg(tmpRaw, afterTransform);
    mat2LabelImg(tmpRaw, canny);
    mat2LabelImg(tmpRaw, contuor);
    mat2LabelImg(tmpRaw, boxed);
    mat2LabelImg(tmpPlate, plate);
    mat2LabelImg(tmpPlate, vEdge);
    mat2LabelImg(tmpPlate, hEdge);

    for (int i = 0; i < 8; i++)
    {
        Mat tmpSeg = Mat::zeros(140, 440 / 7, CV_8UC3);
        mat2LabelImg(tmpSeg, segPic[i]);
    }

    imgLayout1->addWidget(raw, 1, 1);
    imgLayout1->addWidget(afterMask, 1, 2);
    imgLayout1->addWidget(afterTransform, 1, 3);
    imgLayout1->addWidget(canny, 2, 1);
    imgLayout1->addWidget(contuor, 2, 2);
    imgLayout1->addWidget(boxed, 2, 3);
    imgLayout3->addWidget(plate);
    imgLayout3->addWidget(hEdge);
    imgLayout3->addWidget(vEdge);
    imgLayout3->addWidget(res);

    imgLayout2->addLayout(imgLayout1);
    imgLayout2->addLayout(imgLayout3);

    for (auto x : segPic)
        imgLayout5->addWidget(x);

    imgLayout4->addLayout(imgLayout2);
    imgLayout4->addLayout(imgLayout5);
    mainLayout->addLayout(imgLayout4);
    mainLayout->addLayout(btnLayout);

    centerWid->setLayout(mainLayout);
    setCentralWidget(centerWid);

    videoThread = new videoPicThread(raw);
    thread = new QThread;
    connect(thread, &QThread::started, videoThread, &videoPicThread::show);
    videoThread->moveToThread(thread);
}

void Qui::buildconnections()
{
    if (openBtn == nullptr)
    {
        cerr << "openBtn is nullptr" << endl;
        exit(-1);
    }
    connect(openBtn, &QPushButton::clicked, this, &Qui::openImg);

    connect(videoThread, &videoPicThread::readyFrame, this, [this](Mat img)
            { this->rawImg = img; mat2LabelImg(img, this->raw, bigSize) ; });
    connect(saveBtn, &QPushButton::clicked, this, [this]()
            { this->thread->start(); });

    connect(videoThread, &videoPicThread::stopThread, this, [this]()
            { this->thread->quit(); });

    connect(transformBtn, &QPushButton::clicked, this, &Qui::transformImg);

    connect(detectBtn, &QPushButton::clicked, this, &Qui::detectPlate);
}
void Qui::mat2LabelImg(Mat &src, QLabel *dst, Size imsz)
{
    if (!imsz.height)
        imsz = src.size();
    Mat tmp = src.clone();
    if (!tmp.empty())
    {
        cv::resize(tmp, tmp, imsz);
        QImage image;
        if (tmp.type() == CV_8UC3)
        {
            cvtColor(tmp, tmp, COLOR_BGR2RGB);
            image = QImage(tmp.data, tmp.cols, tmp.rows, tmp.step, QImage::Format_RGB888);
        }
        else if (tmp.type() == CV_8UC1)
            image = QImage(tmp.data, tmp.cols, tmp.rows, tmp.step, QImage::Format_Grayscale8);
        QPixmap pixmap = QPixmap::fromImage(image);
        if (!pixmap.isNull())
            dst->setPixmap(pixmap);
    }
}

void Qui::openImg()
{
    {
        QMutexLocker metaLock(&mutex);
        isVideo = false;
    }
    imgPath = QFileDialog::getOpenFileName(this, "打开图片", "", "Image (*.jpg *.png)").toStdString();
    if (!imgPath.empty())
    {
        rawImg = imread(imgPath);
        Mat showImg;
        cv::resize(rawImg, showImg, bigSize);
        if (!showImg.empty())
        {
            mat2LabelImg(showImg, raw, bigSize);
            Mat tmpBoxed = Mat::zeros(320, 240, CV_8UC3);
            Mat tmpPlate = Mat::zeros(70, 220, CV_8UC3);
            mat2LabelImg(tmpBoxed, boxed, bigSize);
            mat2LabelImg(tmpBoxed, contuor, bigSize);
            mat2LabelImg(tmpBoxed, afterMask, bigSize);
            mat2LabelImg(tmpBoxed, afterTransform, bigSize);
            mat2LabelImg(tmpBoxed, canny, bigSize);

            mat2LabelImg(tmpPlate, plate, plateSize);
            mat2LabelImg(tmpPlate, vEdge, plateSize);
            mat2LabelImg(tmpPlate, hEdge, plateSize);
            for (int i = 0; i < 8; i++)
            {
                Mat tmpSeg = Mat::zeros(140, 440 / 7, CV_8UC3);
                mat2LabelImg(tmpSeg, segPic[i], Size(440 / 7, 140));
            }
        }
    }
}
void Qui::transformImg()
{
    if (rawImg.empty())
    {
        QMessageBox::warning(this, "提取车牌", "请先打开图片！");
        return;
    }
    bool flag = false;
    Mat mask, img, hsv, tmp;
    img = rawImg.clone();
    // img.convertTo(img, -1, 1.2, -50); // 1.5是对比度，-50是亮度调整值
    mat2LabelImg(img, raw, bigSize);
    double plateRatioRefer = 140.0f / 440.0f;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
                      -1, 5, -1,
                      0, -1, 0);

    // 应用锐化卷积核
    cv::filter2D(img, img, img.depth(), kernel);
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, rangeLow, rangeHigh, mask);

    mat2LabelImg(mask, afterMask, bigSize);

    /*处理掩膜使之覆盖整个车牌*/
    Mat k_element = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, k_element, Point(-1, -1), 6);
    morphologyEx(mask, mask, MORPH_OPEN, k_element, Point(-1, -1), 6);
    mat2LabelImg(mask, afterTransform, bigSize);

    /*寻找车牌轮廓*/
    Canny(mask, mask, 50, 150);

    mat2LabelImg(mask, canny, bigSize);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    tmp = rawImg.clone();
    drawContours(tmp, contours, -1, Scalar(255, 255, 255), 2);
    mat2LabelImg(tmp, contuor, bigSize);

    RotatedRect roi;
    if (!contours.empty())
    {
        vector<RotatedRect> box;
        for (auto x : contours)
        {
            auto rect = minAreaRect(x);
            double ratio = min(rect.size.width, rect.size.height) / max(rect.size.width, rect.size.height);
            if (ratio > 0.4 * plateRatioRefer && ratio < 1.8 * plateRatioRefer)
                box.push_back(rect);
        }
        if (!box.empty())
        {
            sort(box.begin(), box.end(), [](RotatedRect a, RotatedRect b)
                 { return getRotatedRectArea(a) > getRotatedRectArea(b); });
            roi = box[0];
            Mat afterTrans(Size(440, 140), CV_8UC3);
            vector<Point2f> originPoint;
            getRotatedRectPoints(roi, originPoint);
            Mat perspectiveMatrix = getPerspectiveTransform(originPoint, vec);
            warpPerspective(img, afterTrans, perspectiveMatrix, afterTrans.size());
            drawRotatedRect(tmp, roi);
            mat2LabelImg(tmp, boxed, bigSize);
            mat2LabelImg(afterTrans, plate, plateSize);
            img = afterTrans.clone();

            cvtColor(img, img, COLOR_BGR2GRAY);
            threshold(img, img, 127, 255, THRESH_OTSU);
            img = ~img;
            Mat hProjection, vProjection;
            auto numH = getHorizontalProjection(img, hProjection);
            findHEdge(img, img, numH);
            if (!img.empty())
            {
                mat2LabelImg(img, hEdge, plateSize);
                auto numV = getVerticalProjection(img, vProjection);
                findVEdge(img, img, numV, {0.2, 1.8});
                if (!img.empty())
                {
                    plateImg = img.clone();
                    mat2LabelImg(img, vEdge, plateSize);
                    charSeg(img, segImg, {0.2, 1.8});
                    for (int i = 0; i < segImg.size(); i++)
                    {
                        if (!segImg[i].empty())
                        {
                            mat2LabelImg(segImg[i], segPic[i]);
                            QString plateStr = "";
                            if (!segImg[0].empty())
                            {
                                Mat x_ = segImg[0].clone();
                                x_ = ~x_;
                                cv::resize(x_, x_, Size(20, 20));
                                x_.convertTo(x_, CV_32F);
                                hwc2chw(x_);
                                trtProvince.Infer(x_.data);
                                auto x = trtProvince.get_output_tensor();
                                int idx = argmax(x.data(), x.size());
                                plateStr += regions[idx];
                            }
                            for (int i = 1; i < 7; i++)
                            {
                                if (!segImg[i].empty())
                                {
                                    Mat x_ = segImg[i].clone();
                                    x_ = ~x_;
                                    cv::resize(x_, x_, Size(20, 20));
                                    x_.convertTo(x_, CV_32F);
                                    hwc2chw(x_);
                                    trtNumAlpha.Infer(x_.data);
                                    auto x = trtNumAlpha.get_output_tensor();
                                    int idx = argmax(x.data(), x.size());
                                    plateStr += numalpha[idx];
                                    if (i == 1)
                                        plateStr += "*";
                                }
                            }
                            res->setText(plateStr);
                        }
                        flag = true;
                    }
                }
            }
        }
        if (!flag)
        {
            QMessageBox::warning(this, "提取车牌", "车牌提取失败！");
            return;
        }
    }
}
void Qui::detectPlate()
{
    exit(0);
}
