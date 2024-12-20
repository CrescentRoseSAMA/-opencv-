#include "./seg.hpp"
using namespace cv;
using namespace std;
Mat afterTrans(Size(440, 140), CV_8UC3);
void drawRotatedRect(Mat &img, RotatedRect rect)
{
    Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
}

double getRotatedRectArea(RotatedRect rect)
{
    return rect.size.width * rect.size.height;
}

void getRotatedRectPoints(RotatedRect rect, vector<Point2f> &pts)
{
    Point2f vertices[4];
    Point2f left[2];
    Point2f right[2];
    rect.points(vertices);
    Point2f center = rect.center;
    int n = 0, m = 0;
    for (int i = 0; i < 4; i++)
    {
        if (vertices[i].x - center.x < 0)
            left[n++] = vertices[i];
        else if (vertices[i].x - center.x > 0)
            right[m++] = vertices[i];
    }
    if (left[0].y > left[1].y)
        swap(left[0], left[1]);
    if (right[0].y > right[1].y)
        swap(right[0], right[1]);
    pts.push_back(left[0]);  // 左上
    pts.push_back(left[1]);  // 左下
    pts.push_back(right[1]); // 右下
    pts.push_back(right[0]); // 右上
}
bool findPlate(Mat &src, Mat &dst, Mat &boxed, Mat &maskImg)
{
    Mat mask, img, hsv, tmp;
    double plateRatioRefer = 140.0f / 440.0f;
    img = src.clone();
    tmp = src.clone();
    GaussianBlur(img, img, Size(5, 5), 0);
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, rangeLow, rangeHigh, mask);
    /*处理掩膜使之覆盖整个车牌*/
    Mat k_element = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, k_element, Point(-1, -1), 4);
    morphologyEx(mask, mask, MORPH_ERODE, k_element, Point(-1, -1), 1);
    maskImg = mask.clone();
    /*寻找车牌轮廓*/
    Canny(mask, mask, 50, 150);
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    RotatedRect roi;
    if (!contours.empty())
    {
        vector<RotatedRect> box;
        for (auto x : contours)
        {
            auto rect = minAreaRect(x);
            // double ratio = min(rect.size.width, rect.size.height) / max(rect.size.width, rect.size.height);
            // if (ratio > 0.4 * plateRatioRefer && ratio < 1.2 * plateRatioRefer)
            box.push_back(rect);
        }
        if (!box.empty())
        {
            sort(box.begin(), box.end(), [](RotatedRect a, RotatedRect b)
                 { return getRotatedRectArea(a) > getRotatedRectArea(b); });
            roi = box[0];
        }
        else
            return false;
    }
    else
        return false;

    vector<Point2f> originPoint;
    getRotatedRectPoints(roi, originPoint);
    Mat perspectiveMatrix = getPerspectiveTransform(originPoint, vec);
    warpPerspective(img, afterTrans, perspectiveMatrix, afterTrans.size());
    drawRotatedRect(tmp, roi);
    boxed = tmp.clone();
    dst = afterTrans.clone();
    return true;
}

vector<int> getHorizontalProjection(Mat &src, Mat &dst)
{
    Mat img = src.clone();
    if (img.channels() > 1)
    {
        cvtColor(img, img, COLOR_BGR2GRAY);
        threshold(img, img, 127, 255, THRESH_OTSU);
        img = ~img;
    }
    int col = img.cols, row = img.rows;
    uchar *data = img.data;
    vector<int> numPerRow(row, 0);
    Mat res(img.size(), CV_8UC1, Scalar::all(255));
    uchar *resData = res.data;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            numPerRow[i] += int(data[i * col + j] == 0);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < numPerRow[i]; j++)
            resData[i * col + j] = 0;
    }
    dst = res.clone();
    return numPerRow;
}

vector<int> getVerticalProjection(Mat &src, Mat &dst)
{
    Mat img = src.clone();
    if (img.channels() > 1)
    {
        cvtColor(img, img, COLOR_BGR2GRAY);
        threshold(img, img, 127, 255, THRESH_OTSU);
        img = ~img;
    }
    int col = img.cols, row = img.rows;
    uchar *data = img.data;
    vector<int> numPerCol(col, 0);
    Mat res(img.size(), CV_8UC1, Scalar::all(255));
    uchar *resData = res.data;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
            numPerCol[i] += int(data[i + j * col] == 0);
    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < numPerCol[i]; j++)
            resData[i + j * col] = 0;
    }
    flip(res, res, 0);
    dst = res.clone();
    return numPerCol;
}

void findHEdge(Mat &src, Mat &dst, vector<int> perRow)
{
    Mat img = src.clone();
    double resMean = cv::mean(perRow)[0];
    int resMin = *std::min_element(perRow.begin(), perRow.end());
    double resRefer = (resMean + resMin) / 2;
    int col = img.cols, row = img.rows;
    // 先找上下边界
    int tmpTop = 0;
    int tmpBottom = 0;
    int maxLen = 0;
    int top = 0;
    int bottom = 0;
    for (int i = 0; i < row; i++)
    {
        if (perRow[i] >= resRefer)
            tmpBottom += 1;
        else
        {
            if (tmpBottom - tmpTop > maxLen)
            {
                top = tmpTop;
                bottom = tmpBottom;
                maxLen = bottom - top;
            }
            tmpTop = i;
            tmpBottom = tmpTop;
        }
    }
    Rect roi(0, top, col, bottom - top);
    dst = img(roi).clone();
}

void findVEdge(Mat &src, Mat &dst, vector<int> perCol, pair<double, double> charThres)
{
    Mat img = src.clone();
    double resMean = cv::mean(perCol)[0];
    int resMin = *std::min_element(perCol.begin(), perCol.end());
    double resRefer = (resMean + resMin) / 1.5;
    int col = img.cols, row = img.rows;
    double charLenRefer = 45.0f / 440.0f * col;

    /*找左边界*/
    int tmpLeft = 0;
    int tmpRight = 0;
    int left = 0;
    int right = 0;
    for (int i = 0; i < col; i++)
    {
        if (perCol[i] >= resRefer)
            tmpRight += 1;
        else
        {
            double charLen = tmpRight - tmpLeft;
            double ratio = min(charLen, charLenRefer) / max(charLen, charLenRefer);
            if (ratio > charThres.first && ratio < charThres.second)
            {

                left = tmpLeft;
                break;
            }
            tmpRight = i;
            tmpLeft = tmpRight;
        }
    }
    /*找右边界*/
    tmpRight = col;
    tmpLeft = tmpRight;
    for (int i = col - 1; i >= 0; i--)
    {
        if (perCol[i] >= resRefer)
            tmpLeft -= 1;
        else
        {
            int charLen = tmpRight - tmpLeft;
            double ratio = (double)charLen / charLenRefer;
            if (ratio > charThres.first && ratio < charThres.second)
            {
                right = tmpRight;
                break;
            }
            tmpLeft = i;
            tmpRight = tmpLeft;
        }
    }
    Rect roi(left, 0, right - left, row);
    dst = img(roi).clone();
}

bool findPlateEdge(Mat &src, Mat &dst, Mat &boxed, Mat &maskImg)
{
    Mat img = src.clone();
    bool flag = findPlate(img, img, boxed, maskImg);
    if (!flag)
        return false;

    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
                      -1, 5, -1,
                      0, -1, 0);

    // 应用锐化卷积核
    cv::filter2D(img, img, img.depth(), kernel);
    cvtColor(img, img, COLOR_BGR2GRAY);
    threshold(img, img, 127, 255, THRESH_OTSU);
    img = ~img;
    Mat hProjection, vProjection;
    auto numH = getHorizontalProjection(img, hProjection);
    findHEdge(img, img, numH);

    auto numV = getVerticalProjection(img, vProjection);
    findVEdge(img, img, numV, {0.2, 1.8});
    resize(img, img, Size(440, 140));
    dst = img.clone();
    return true;
}

void charSeg(Mat &src, vector<Mat> &dst, pair<double, double> charThres)
{

    dst.clear();
    Mat img = src.clone();
    Mat tmp;
    auto perCol = getVerticalProjection(img, tmp);
    double resMean = cv::mean(perCol)[0];
    int resMin = *std::min_element(perCol.begin(), perCol.end());
    double resRefer = (resMean + resMin) / 4;

    int tmpRight = 0;
    int tmpLeft = 0;
    int row = img.rows, col = img.cols;
    double charLenRefer = 45.0f / 440.0f * col;
    for (int i = 0; i < col; i++)
    {
        if (perCol[i] > resRefer)
            tmpRight += 1;
        else
        {
            tmpLeft += 1;
            double charLen = tmpRight - tmpLeft;
            double ratio = min(charLen, charLenRefer) / max(charLen, charLenRefer);
            if (ratio > charThres.first && ratio < charThres.second)
            {
                // if (charLen <= 0.8 * charLenRefer)
                // {
                //     tmpRight += charLenRefer * 0.8 / 3.5;
                //     tmpLeft = tmpLeft - charLenRefer * 0.8 / 3.5;
                // }
                Mat charImg = img(Rect(tmpLeft, 0, tmpRight - tmpLeft, row));
                dst.push_back(charImg);
            }
            tmpRight = i;
            tmpLeft = tmpRight;
        }
        if (i == col - 1)
        {
            double charLen = tmpRight - tmpLeft;
            double ratio = min(charLen, charLenRefer) / max(charLen, charLenRefer);
            if (ratio > charThres.first && ratio < charThres.second)
            {
                Mat charImg = img(Rect(tmpLeft, 0, tmpRight - tmpLeft, row));
                dst.push_back(charImg);
            }
        }
    }
}

// bool runSeg(Mat &src, Mat &plate, Mat &boxed, vector<Mat> &chars, pair<double, double> charThres)
// {

//     bool flag = findPlateEdge(src, plate, boxed);
//     if (!flag)
//         return false;
//     charSeg(plate, chars, charThres);
//     return true;
// }