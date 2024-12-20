#ifndef SEG_HPP
#define SEG_HPP
#include <opencv4/opencv2/opencv.hpp>
const cv::Scalar rangeLow{32, 58, 151};
const cv::Scalar rangeHigh{180, 255, 255};
const std::vector<cv::Point2f> vec{
    cv::Point2f(0, 0),     // 左上
    cv::Point2f(0, 140),   // 左下
    cv::Point2f(440, 140), // 右下
    cv::Point2f(440, 0),   // 右上
};
double getRotatedRectArea(cv::RotatedRect rect);
void drawRotatedRect(cv::Mat &img, cv::RotatedRect rect);
void getRotatedRectPoints(cv::RotatedRect rect, std::vector<cv::Point2f> &pts);
bool findPlate(cv::Mat &src, cv::Mat &dst, cv::Mat &boxed, cv::Mat &maskImg);
std::vector<int> getHorizontalProjection(cv::Mat &src, cv::Mat &dst);
std::vector<int> getVerticalProjection(cv::Mat &src, cv::Mat &dst);
void findHEdge(cv::Mat &src, cv::Mat &dst, std::vector<int> perRow);
void findVEdge(cv::Mat &src, cv::Mat &dst, std::vector<int> perCol, std::pair<double, double> charThres = {0.8, 1.2});
bool findPlateEdge(cv::Mat &src, cv::Mat &dst, cv::Mat &maskImg);
void charSeg(cv::Mat &src, std::vector<cv::Mat> &dst, std::pair<double, double> charThres = {0.8, 1.2});
// bool runSeg(cv::Mat &src, cv::Mat &plate, cv::Mat &boxed, std::vector<cv::Mat> &chars, std::pair<double, double> charThres = {0.8, 1.2});
#endif // SEG_HPP