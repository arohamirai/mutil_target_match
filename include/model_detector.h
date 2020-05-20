#ifndef MODEL_DETECTOR_H
#define MODEL_DETECTOR_H

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
class AwesomeClock
{
  public:
  AwesomeClock();
  ~AwesomeClock();
  void start();

  // return time elapsed(seconds) from start() called.
  double stop();

  private:
  std::chrono::steady_clock::time_point t_point_;
  std::chrono::duration<double> time_elapse_;
};

typedef struct _Pose2D
{
	double x;
	double y;
	double theta;
} Pose2D;

class ModelDetector
{
  public:
  typedef struct _MatchResult
  {
    bool invalid;
    cv::Point2d best_loc;
    double best_angle;
    double best_value;
    Pose2D model_in_map;
    cv::Point2f corners[4];
    cv::Point2f center;
    cv::Mat cur_axes;
  } MatchResult;

  public:
  ModelDetector(bool enable_openmp = true);
  ~ModelDetector();


  std::vector<MatchResult>
  pyramidMatching(const cv::Mat& src, const cv::Mat& templ,
                  std::string& error_string, double thresh = 0.7,
                  int pyramid_layers = 3,
                  int match_method = CV_TM_CCORR_NORMED);

  void drawMatch(cv::Mat& img_show, const cv::Mat& model,
                 std::vector<MatchResult>& match_results);

  private:
  bool roughMatch(const cv::Mat& src, const cv::Mat& templ, double thresh,
                  std::vector<MatchResult>& cand_results, int match_method);
  bool matchingInLayer(const cv::Mat& src, const cv::Mat& templ,
                       int match_method, cv::Point& best_loc,
                       double& best_angle, double& best_value,
                       std::string& error_string,
                       cv::Point guess_loc = cv::Point(0, 0),
                       double guess_angle = 0., double angle_search_max = 180.,
                       double angle_step = 0.1);
  void pixResult2PhyResult(std::vector<MatchResult>& match_results,
                           cv::Size2f model_size);

  float resolution_;
  float min_corner_[2], max_corner_[2];
  bool enable_openmp_;
  bool debug_;
  AwesomeClock clock_;

  std::string record_path_prefix_;
};

#endif
