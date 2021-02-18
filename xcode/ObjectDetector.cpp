//
//  ObjectDetector.cpp
//  FaceDetector
//
//  Created by Fabian Töpfer on 09.02.21.
//  Copyright © 2021 baniaf. All rights reserved.
//

#include "ObjectDetector.hpp"
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include<fstream>

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "Resources.h"

ObjectDetector::ObjectDetector():
m_confidence_threshold(0.5),
m_input_image_height(300),
m_input_image_width(300),
m_scale_factor(1.0),
m_mean_values({104., 177.0, 123.0}){
    
        // get labels of all classes
        
    
        std::string classesFile = ci::app::getResourcePath(COCO_NAMES).string();
        std::ifstream ifs(classesFile.c_str());
        std::string line;
        while (getline(ifs, line)) classes.push_back(line);
    
        std::cout<<"Classes: "<<classes.size()<<std::endl;
        
        std::string cfgFile = ci::app::getResourcePath(YOLO_CFG).string();
        std::string weightsFile = ci::app::getResourcePath(YOLO_WEIGHTS).string();
    
        m_network = cv::dnn::readNetFromDarknet(cfgFile,weightsFile);
        m_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    
    
    
        if (m_network.empty()) {
            std::ostringstream ss;
            ss << "Failed to load network with the following settings:\n";
            throw std::invalid_argument(ss.str());
        }
       
        auto layers = m_network.getLayerNames();
        auto i = m_network.getUnconnectedOutLayers();
        m_outputLayer = layers[i[0] - 1];
}

std::vector<cv::Rect> ObjectDetector::detect_object_rects(cv::UMat &frame) {
    
    cv::Mat input_blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(416, 416), false, false);
    
   
    m_network.setInput(input_blob, "data");
    
    std::vector<cv::Mat> detection;
    detection.clear();
    m_network.forward(detection,getOutputsNames(m_network));
    
    std::vector<cv::Rect> boxes = remove_box(frame, detection);
    
//    cv::Mat detectedFrame;
//    frame.convertTo(detectedFrame, CV_8U);
//    static const std::string kWinName = "Deep learning object detection in OpenCV";
//
//    imshow(kWinName, frame);
    
    return boxes;
    
}
std::vector<std::string> ObjectDetector::getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void ObjectDetector::draw(){
    
};

std::vector<cv::Rect>  ObjectDetector::remove_box(cv::UMat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            
            if (confidence > 0.4)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        draw_box(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
    
    return boxes;
}

void ObjectDetector::draw_box(int classId, float conf, int left, int top, int right, int bottom, cv::UMat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}
