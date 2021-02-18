//
//  ObjectDetector.hpp
//  FaceDetector
//
//  Created by Fabian Töpfer on 09.02.21.
//  Copyright © 2021 baniaf. All rights reserved.
//

#ifndef ObjectDetector_hpp
#define ObjectDetector_hpp

#include <stdio.h>
#include <opencv2/dnn.hpp>

class ObjectDetector{
public:
    explicit ObjectDetector();
    
    std::vector<cv::Rect> detect_object_rects(cv::UMat& frame);
    
    void draw();
    
private:
    const int m_input_image_width;
    const int m_input_image_height;
    const double m_scale_factor;
    const cv::Scalar m_mean_values;
    const float m_confidence_threshold;
    
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
    std::vector<cv::Rect>  remove_box(cv::UMat& frame, const std::vector<cv::Mat>& outs);
    void draw_box(int classId, float conf, int left, int top, int right, int bottom, cv::UMat& frame);

    cv::dnn::Net m_network;
    std::string m_outputLayer;
    std::vector<std::string> classes;
};
#endif /* ObjectDetector_hpp */
