#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "CinderOpenCV.h"
#include "ObjectDetector.hpp"
#include <opencv2/videoio.hpp>

using namespace ci;
using namespace ci::app;
using namespace std;

class ObjectDetector_cinderApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
    
private:
    ObjectDetector m_detector;
    cv::VideoCapture m_camera;
    cv::Mat m_detectionFrame;
};

void ObjectDetector_cinderApp::setup()
{
    m_camera = cv::VideoCapture(0);
    if(m_camera.isOpened()){
        m_camera >> m_detectionFrame;
    };
    
}

void ObjectDetector_cinderApp::mouseDown( MouseEvent event )
{
}

void ObjectDetector_cinderApp::update()
{
    if(m_camera.isOpened()){
        m_camera >> m_detectionFrame;
    
        cv::UMat umat;
        m_detectionFrame.copyTo(umat);
        m_detector.detect_object_rects(umat);
        umat.copyTo(m_detectionFrame);
    };
}

void ObjectDetector_cinderApp::draw()
{
    ImageSourceRef imgRef =  fromOcv(m_detectionFrame);
    gl::Texture2dRef tex = gl::Texture2d::create(imgRef);
    gl::draw(tex);
}

CINDER_APP( ObjectDetector_cinderApp, RendererGl )
