#include "qui.hpp"
#include <QApplication>

int main(int argc, char *argv[])
{
    qRegisterMetaType<cv::Mat>("cv::Mat&");
    qRegisterMetaType<cv::Mat>("cv::Mat");
    QApplication app(argc, argv);
    Qui qui;
    qui.show();

    return app.exec();
}