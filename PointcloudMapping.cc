
#include "PointcloudMapping.h"

namespace ORB_SLAM2 {

PointCloudMapping::PointCloudMapping(double resolution)
{
    mResolution = resolution;
    mCx = 0;
    mCy = 0;
    mFx = 0;
    mFy = 0;
    mbShutdown = false;
    mbFinish = false;

     //滤波器
    voxel.setLeafSize( resolution, resolution, resolution);
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);

    //Point_Map & octree
    mPointCloud = boost::make_shared<PointCloud>(); 
    tree = new octomap::ColorOcTree(0.01);

    //建图线程
    viewerThread = std::make_shared<std::thread>(&PointCloudMapping::showPointCloud, this); 
}



PointCloudMapping::~PointCloudMapping()
{
    viewerThread->join();
}


void PointCloudMapping::requestFinish()
{
    {
        unique_lock<mutex> locker(mKeyFrameMtx);
        mbShutdown = true;
    }
    mKeyFrameUpdatedCond.notify_one();
}

bool PointCloudMapping::isFinished()
{
    return mbFinish;
}


// Tracking 调用 ( KeyFrame* ， const cv::Mat&  ,const cv::Mat&) kf color depth 压入队列
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth)
{
    unique_lock<mutex> locker(mKeyFrameMtx);
    
    mvKeyFrames.push(kf);
    mvColorImgs.push( color.clone() );  
    mvDepthImgs.push( depth.clone() );

    mKeyFrameUpdatedCond.notify_one();
    cout << "receive a keyframe, id = " << kf->mnId << endl;
}


void PointCloudMapping::showPointCloud() 
{
	//pcl_viewer
    pcl::visualization::CloudViewer viewer("Dense pointcloud viewer");
   
   	
    while(true) {   
        KeyFrame* kf;
        cv::Mat colorImg, depthImg;
        
        {
            std::unique_lock<std::mutex> locker(mKeyFrameMtx);
            
            while(mvKeyFrames.empty() && !mbShutdown)
            {  
                mKeyFrameUpdatedCond.wait(locker); 
            }            
            
            if (!(mvDepthImgs.size() == mvColorImgs.size() && mvKeyFrames.size() == mvColorImgs.size())) {
                std::cout << "Unexpect ！" << std::endl;
                continue;
            }

            if (mbShutdown && mvColorImgs.empty() && mvDepthImgs.empty() && mvKeyFrames.empty()) {
                break;
            }
		//从队列取出局部点云信息 KF color Depth
            kf = mvKeyFrames.front();
            colorImg = mvColorImgs.front();    
            depthImg = mvDepthImgs.front();    
            mvKeyFrames.pop();
            mvColorImgs.pop();
            mvDepthImgs.pop();
        }

        if (mCx==0 || mCy==0 || mFx==0 || mFy==0) {
            mCx = kf->cx;
            mCy = kf->cy;
            mFx = kf->fx;
            mFy = kf->fy;
        }

        
        {
            std::unique_lock<std::mutex> locker(mPointCloudMtx);
            
        
            //生成局部点云  插入全局点云  和  octomap中
            generatePointCloud(colorImg, depthImg, kf->GetPose(), kf->mnId);
            
            
            
            viewer.showCloud(mPointCloud);
            
        }
        std::cout << "show point cloud, size=" << mPointCloud->points.size() << std::endl;
    }

    // 存储点云
    string save_path = "./resultPointCloudFile.pcd";
    pcl::io::savePCDFile(save_path, *mPointCloud);
    
    tree->write("octomap.ot");
    
    cout << "save pcd files to :  " << save_path << endl;
    mbFinish = true;
}


//
//注意d控制深度值
void PointCloudMapping::generatePointCloud(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& pose, int nId)
{ 


    std::cout << "Converting image: " << nId;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();   
    
    Eigen::Isometry3d T = Converter::toSE3 Quat( pose );
 
 
    //
    PointCloud::Ptr current(new PointCloud);

    for(size_t v = 0; v < imRGB.rows ; v+=3){
        for(size_t u = 0; u < imRGB.cols ; u+=3){
            float d = imD.ptr<float>(v)[u];
            if(d <0.01 || d>15){ // 深度值为0 表示测量失败
                continue;
            }

            PointT p;
            Eigen::Vector3d p1;
            
            p.z = d;
            p.x = ( u - mCx) * p.z / mFx;
            p.y = ( v - mCy) * p.z / mFy;
            
            p.b = imRGB.ptr<uchar>(v)[u*3];
            p.g = imRGB.ptr<uchar>(v)[u*3+1];
            p.r = imRGB.ptr<uchar>(v)[u*3+2];
		
            current->points.push_back(p);
            
        }        
    }

    
    PointCloud::Ptr tmp(new PointCloud);
    
    
    // tmp为转换到世界坐标系下的点云
    pcl::  (*current, *tmp, T.inverse().matrix()); 

    //   T * P_world  = P_camera；
     //    P_world = P_camera * T.inverse().matrix();
     //   T    :     ( R T
     //                0 1 )           t(0,3) T(1,3) t(2,3)  (X,Y,Z)

   
   //可以做一些滤波   Not necessary   Example
  //  voxel.setInputCloud(tmp);
  //  voxel.filter(*current);
  
     current=tmp;
     current->is_dense = true; 
 
 
   //插入octomap
     for (auto p:current->points)
    {
         // 将点云里的点插入到octomap中
         tree->updateNode( octomap::point3d(p.x, p.y, p.z), true );
     }
 
     	// 设置颜色
     for (auto p:current->points)
     {
         tree->integrateNodeColor( p.x, p.y, p.z, p.r, p.g, p.b );
     }
     	// 更新octomap
     tree->updateInnerOccupancy();
     
     
    //插入全局点云
    *mPointCloud += *current;

 
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); 
    std::cout << ", Cost = " << t << std::endl;
}


void PointCloudMapping::getGlobalCloudMap(PointCloud::Ptr &outputMap)
{
    std::unique_lock<std::mutex> locker(mPointCloudMtx);
    outputMap = mPointCloud;
}



}
