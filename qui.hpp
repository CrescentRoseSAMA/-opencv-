#ifndef QUI_HPP
#define QUI_HPP
#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QLayout>
#include <QMessageBox>
#include <QFileDialog>
#include <QMutex>
#include <QLabel>
#include <QMessageBox>
#include <QThread>
#include <opencv4/opencv2/opencv.hpp>
#include "./charseg/seg.hpp"
#include "./Infer/TRTFrame.hpp"
#include "./MvCamera/MvCamera.h"
extern bool isVideo;
class videoPicThread;
class Qui : public QMainWindow
{
    Q_OBJECT
public:
    QMutex mutex;
    Qui(QWidget *parent = nullptr);
    void creatLayout();
    void buildconnections();
    void mat2LabelImg(cv::Mat &src, QLabel *dst, cv::Size imsz = cv::Size(0, 0));

private slots:
    void openImg();
    void transformImg();
    void detectPlate();

private:
    QThread *thread;
    QWidget *centerWid;
    QVBoxLayout *mainLayout;
    QHBoxLayout *btnLayout;
    QGridLayout *imgLayout1;
    QVBoxLayout *imgLayout3;
    QHBoxLayout *imgLayout2;
    QVBoxLayout *imgLayout4;
    QHBoxLayout *imgLayout5;
    QPushButton *openBtn;
    QPushButton *saveBtn;
    QPushButton *transformBtn;
    QPushButton *detectBtn;
    videoPicThread *videoThread;
    QLabel *raw;
    QLabel *afterMask;
    QLabel *afterTransform;
    QLabel *canny;
    QLabel *contuor;
    QLabel *boxed;

    QLabel *plate;
    QLabel *vEdge;
    QLabel *hEdge;
    QLabel *res;
    std::vector<QLabel *> segPic{8};

    cv::Mat rawImg;
    cv::Mat plateImg;
    std::vector<cv::Mat> segImg{8};

    std::string imgPath;

    TRTFrame trtProvince;
    TRTFrame trtNumAlpha;
};

// 失败
class videoPicThread : public QObject
{
    Q_OBJECT
private:
    bool imgGeted;
    cv::Mat showImg;
    QLabel *imgLabel;
    QMutex mutex;

signals:
    void readyFrame(cv::Mat img);
    void stopThread();

public:
    videoPicThread(QLabel *label, QObject *parent = nullptr) : QObject(parent)
    {
        imgLabel = label;
        imgGeted = false;
        {
            QMutexLocker locker(&mutex);
            isVideo = false;
        }
    }
    void show()
    {
        isVideo = true;
        cv::VideoCapture cap(0);
        if (cap.isOpened())
        {
            cv::Mat tmp;
            while (cap.isOpened() && isVideo)
            {
                cap >> tmp;
                if (tmp.empty())
                    break;
                emit readyFrame(tmp);
                QThread::msleep(10);
            }
        }
        else
        {
            QMessageBox::warning(nullptr, "Warning", "无法USB相机尝试使用MV USB相机");
            Mv_Camera cam;
            bool flag = cam.Open_Camera();
            if (flag)
            {
                while (isVideo)
                {
                    cv::Mat tmp;
                    cam.read(tmp);
                    if (tmp.empty())
                        break;
                    emit readyFrame(tmp);
                    QThread::msleep(10);
                }
            }
            else
            {
                QMessageBox::warning(nullptr, "Warning", "无法打开相机,播放预设视频");
                cap = cv::VideoCapture("../Armor.mp4");
                if (cap.isOpened())
                {
                    cv::Mat tmp;
                    while (cap.isOpened() && isVideo)
                    {
                        cap >> tmp;
                        if (tmp.empty())
                            break;
                        emit readyFrame(tmp);
                        QThread::msleep(10);
                    }
                }
            }
            emit stopThread();
        }
    }
};
#endif // QUI_HPP