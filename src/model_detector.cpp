#include "model_detector.h"
#include <fstream>
using namespace cv;
ModelDetector::ModelDetector(bool enable_openmp)
	:enable_openmp_(enable_openmp)
	, record_path_prefix_ ("./")
{
}
ModelDetector::~ModelDetector()
{
}
bool ModelDetector::roughMatch(const cv::Mat& src, const cv::Mat& templ,
                               double thresh,
                               std::vector<MatchResult>& cand_results,
                               int match_method)
{
  Mat result_score = Mat::zeros(src.size(), CV_32FC1);
  Mat result_angle = Mat::zeros(src.size(), CV_32FC1);

  Mat result_bin = Mat::zeros(src.size(), CV_8UC1);

  Point2f templ_center = Point2f(templ.cols / 2, templ.rows / 2);

  int result_cols = 0;
  int result_rows = 0;
  for (int angle = -180; angle < 180; angle += 1)
  {
    // std::cout << "rough match search angle:" << angle << std::endl;
    // calculate the size of the finsal best bbox
    Mat rot_mat = getRotationMatrix2D(templ_center, angle, 1.);
    cv::Rect rot_bbox =
      cv::RotatedRect(templ_center, templ.size(), angle).boundingRect();

    //  make sure that rotated templ will within the templ_warpped
    rot_mat.at<double>(0, 2) += rot_bbox.width / 2.0 - templ_center.x;
    rot_mat.at<double>(1, 2) += rot_bbox.height / 2.0 - templ_center.y;

    Mat templ_warpped;
    cv::warpAffine(templ, templ_warpped, rot_mat, rot_bbox.size());

    // check
    if ((src.cols < templ_warpped.cols) || (src.rows < templ_warpped.rows))
    {
      std::string error_string = "ROI is too small.";
      std::cout << error_string << " imageROI Size:" << src.size()
                << ", templ_warpped Size:" << templ_warpped.size();
      return false;
    }

    result_cols = src.cols - templ_warpped.cols + 1;
    result_rows = src.rows - templ_warpped.rows + 1;
    Mat result = Mat::zeros(result_cols, result_rows, CV_32FC1);

    // Mat mask =
    cv::matchTemplate(src, templ_warpped, result, match_method);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    // std::cout << "minVal: " << minVal << ", maxVal: " << maxVal << ",
    // thresh_value:" << thresh << std::endl;
    if(match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
    {

      if (minVal > thresh)
        continue;
    }
    else
    {
      if (maxVal < thresh)
        continue;
    }

    for (int r = 0; r < result.rows; r++)
      for (int c = 0; c < result.cols; c++)
      {
        Point p = cv::Point(c, r);
        float value = result.at<float>(p);

        if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
        {
          if (value < thresh)
          {
            result_bin.at<uchar>(p) = 255;

            float pre_best_value = result_score.at<float>(p);
            if (value < pre_best_value)
            {
              result_score.at<float>(p) = value;
              result_angle.at<float>(p) = angle;
            }
          }
        }
        else
        {
          if (value > thresh)
          {
            result_bin.at<uchar>(p) = 255;
            float pre_best_value = result_score.at<float>(p);
            if (value > pre_best_value)
            {
              result_score.at<float>(p) = value;
              result_angle.at<float>(p) = angle;
            }
          }
        }
      }
  }

  Mat labels_img;
  int nccomps = connectedComponents(result_bin, labels_img, 8, CV_32S);

  // index zero is background
  for (int r = 1; r < nccomps; r++)
  {
    Mat mask_img = (labels_img == r);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    ;
    cv::minMaxLoc(result_score, &minVal, &maxVal, &minLoc, &maxLoc, mask_img);

    Point best_loc;

    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
    {
      best_loc = minLoc;
    }
    else
    {
      best_loc = maxLoc;
    }
    MatchResult candidate_result;
    candidate_result.invalid = false;
    candidate_result.best_angle = result_angle.at<float>(best_loc);
    candidate_result.best_value = result_score.at<float>(best_loc);
    candidate_result.best_loc = best_loc;
    cand_results.push_back(candidate_result);
  }

  return true;
}

std::vector<ModelDetector::MatchResult>
ModelDetector::pyramidMatching(const cv::Mat& src, const cv::Mat& templ,
                               std::string& error_string, double thresh,
                               int pyramid_layers, int match_method)
{
  std::vector<MatchResult> match_results;
  error_string = "";

  Mat src_expand = src;
  Mat templ_expand = templ;
  int bord_width, src_bord_width, templ_bord_width;
  int src_expand_ratio = 3;
  int templ_expand_ratio = 1;

  bord_width = std::pow(2, pyramid_layers - 1);
  src_bord_width = src_expand_ratio * bord_width;
  templ_bord_width = templ_expand_ratio * bord_width;

  if (pyramid_layers > 4)
  {
    copyMakeBorder(src, src_expand, src_bord_width, src_bord_width,
                   src_bord_width, src_bord_width, BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    copyMakeBorder(templ, templ_expand, templ_bord_width, templ_bord_width,
                   templ_bord_width, templ_bord_width, BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));

    int smooth_radius = 2 * bord_width + 1;
    GaussianBlur(src_expand, src_expand, cv::Size(smooth_radius, smooth_radius),
                 0);
    GaussianBlur(templ_expand, templ_expand,
                 cv::Size(smooth_radius, smooth_radius), 0);

    cv::normalize(src_expand, src_expand, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(templ_expand, templ_expand, 0, 255, NORM_MINMAX, CV_8UC1);
    if (debug_)
    {
      imwrite(record_path_prefix_ + "src.bmp", src);
      imwrite(record_path_prefix_ + "templ.bmp", templ);
      imwrite(record_path_prefix_ + "src_expand.bmp", src_expand);
      imwrite(record_path_prefix_ + "templ_expand.bmp", templ_expand);
    }
  }

  std::vector<cv::Mat> pyramid_src_imgs, pyramid_templ_imgs;

  pyramid_src_imgs.push_back(src_expand);
  pyramid_templ_imgs.push_back(templ_expand);

  for (int i = 2; i <= pyramid_layers; i++)
  {
    Mat pyramid_src_img, pyramid_templ_img;
    pyrDown(pyramid_src_imgs.back(), pyramid_src_img);
    pyrDown(pyramid_templ_imgs.back(), pyramid_templ_img);
    pyramid_src_imgs.push_back(pyramid_src_img);
    pyramid_templ_imgs.push_back(pyramid_templ_img);
  }

  bool match_result = true;
  for (int i = pyramid_layers; i > 0; i--)
  {
    clock_.start();
    double thresh_in_layer;
    if (match_method == CV_TM_CCORR_NORMED
        || match_method == CV_TM_SQDIFF_NORMED
        || match_method == CV_TM_CCOEFF_NORMED)
    {
      thresh_in_layer = thresh;
    }
    else
    {
      thresh_in_layer = thresh / std::pow(2, i - 1);
    }

    if (i == pyramid_layers)
    {
      roughMatch(pyramid_src_imgs.at(i - 1), pyramid_templ_imgs.at(i - 1),
                 thresh_in_layer, match_results, match_method);

      for (auto cand_index = 0; cand_index < match_results.size(); cand_index++)
      {
        MatchResult cand_result = match_results[cand_index];
        cand_result.best_loc = cand_result.best_loc * std::pow(2, i - 1);
        match_results[cand_index] = cand_result;
        std::cout << "layer: " << i << ", cand_result_index: " << cand_index
                  << ", best_loc: " << cand_result.best_loc
                  << ",  best_angle: " << cand_result.best_angle
                  << ", best_value: " << cand_result.best_value << std::endl;
      }
    }
    else
    {
#pragma omp parallel for if (enable_openmp_)
      for (auto cand_index = 0; cand_index < match_results.size(); cand_index++)
      {
        cv::Point guess_loc_in_layers;
        cv::Point best_loc_in_layers;
        double best_angle;
        double best_value;
        double guess_angle;

        double angle_search_max_in_layer;
        double angle_step;
        if (i == 1)
        {
          angle_search_max_in_layer = 0.5;
          angle_step = 0.1;
        }
        else
        {
          angle_search_max_in_layer = 2;
          angle_step = 0.2;
        }
        MatchResult& cand_result = match_results[cand_index];

        if (cand_result.invalid)
          continue;

        guess_loc_in_layers = cand_result.best_loc / std::pow(2, i - 1);
        guess_angle = cand_result.best_angle;
        match_result = matchingInLayer(
          pyramid_src_imgs.at(i - 1), pyramid_templ_imgs.at(i - 1),
          match_method, best_loc_in_layers, best_angle, best_value,
          error_string, guess_loc_in_layers, guess_angle,
          angle_search_max_in_layer, angle_step);

        cand_result.best_loc = best_loc_in_layers * std::pow(2, i - 1);
        cand_result.best_angle = best_angle;
        cand_result.best_value = best_value;

        if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
        {
          if (best_value > thresh_in_layer)
            if (best_value  > thresh_in_layer)
              cand_result.invalid = true;
        }
        else
        {
          if (best_value < thresh_in_layer)
            if (best_value  < thresh_in_layer)
              cand_result.invalid = true;
        }
        std::cout << "layer: " << i << ", cand_result_index: " << cand_index
                  << ", best_loc: " << cand_result.best_loc
                  << ",  best_angle: " << cand_result.best_angle
                  << ", best_value: " << cand_result.best_value << std::endl;
      }
    }
    if (!match_result)
      break;
    std::cout << "layer: " << i
              << ", pyramid_src_imgs.size:" << pyramid_src_imgs.at(i - 1).size()
              << ", pyramid_templ_imgs.size:"
              << pyramid_templ_imgs.at(i - 1).size()
              << ", time comsume: " << clock_.stop() << "s." << std::endl;

    if (debug_)
    {
      Mat src_draw = pyramid_src_imgs.at(0).clone();
      Mat model_draw = pyramid_templ_imgs.at(0).clone();
      drawMatch(src_draw, model_draw, match_results);

      std::string iamge_save_name =
        "match_result_layer_" + std::to_string(i) + ".bmp";
      imwrite(record_path_prefix_ + iamge_save_name, src_draw);
    }
  }

  if (pyramid_layers > 4)
  {
    for (auto& result : match_results)
    {
      const Mat newTempl = pyramid_templ_imgs.at(0);
      Point2f newTempl_center =
        Point2f(newTempl.cols / 2.f, newTempl.rows / 2.f);
      float best_angle = result.best_angle;

      cv::Rect rot_bbox_origin =
        cv::RotatedRect(newTempl_center, templ.size(), best_angle)
          .boundingRect();
      cv::Rect rot_bbox_expand_border =
        cv::RotatedRect(newTempl_center, newTempl.size(), best_angle)
          .boundingRect();

      result.best_loc.x +=
        (rot_bbox_expand_border.width - rot_bbox_origin.width) / 2.
        - src_bord_width;
      result.best_loc.y +=
        (rot_bbox_expand_border.height - rot_bbox_origin.height) / 2.
        - src_bord_width;
    }
  }

  std::vector<MatchResult> match_results_filt;
  for (auto result : match_results)
  {
    if (!result.invalid)
    {
      match_results_filt.push_back(result);
    }
  }

  return match_results_filt;
}

bool ModelDetector::matchingInLayer(const cv::Mat& src, const cv::Mat& templ,
                                    int match_method, cv::Point& best_loc,
                                    double& best_angle, double& best_value,
                                    std::string& error_string,
                                    cv::Point guess_loc, double guess_angle,
                                    double angle_search_max, double angle_step)
{
  best_value = -1;
  best_angle = 0.;
  Mat imageROI;
  cv::Point imageROI_origin_in_src;
  cv::Point2f templ_center;
  int margin;

  templ_center = Point2f(templ.cols / 2, templ.rows / 2);

  // maybe 5 is small
  margin = 11;
  cv::Rect max_bbox_py =
    cv::RotatedRect(templ_center, templ.size(), guess_angle).boundingRect();

  // check
  int roi_lx = (guess_loc.x - margin) > 0 ? (guess_loc.x - margin) : 0;
  int roi_ly = (guess_loc.y - margin) > 0 ? (guess_loc.y - margin) : 0;
  int roi_w =
    ((guess_loc.x + max_bbox_py.width + margin) < src.cols && roi_lx > 0)
      ? (max_bbox_py.width + 2 * margin)
      : (src.cols - roi_lx);
  int roi_h =
    ((guess_loc.y + max_bbox_py.height + margin) < src.rows && roi_ly > 0)
      ? (max_bbox_py.height + 2 * margin)
      : (src.rows - roi_ly);
  imageROI = src(Rect(Rect(roi_lx, roi_ly, roi_w, roi_h)));
  imageROI_origin_in_src = cv::Point(roi_lx, roi_ly);

  Mat result;
  int result_cols = 0;
  int result_rows = 0;
  for (double angle = guess_angle - angle_search_max;
       angle < guess_angle + angle_search_max; angle += angle_step)
  {
    // calculate the size of the finsal best bbox
    Mat rot_mat = getRotationMatrix2D(templ_center, angle, 1.);
    cv::Rect rot_bbox =
      cv::RotatedRect(templ_center, templ.size(), angle).boundingRect();

    //  make sure that rotated templ will within the templ_warpped
    rot_mat.at<double>(0, 2) += rot_bbox.width / 2.0 - templ_center.x;
    rot_mat.at<double>(1, 2) += rot_bbox.height / 2.0 - templ_center.y;

    Mat templ_warpped;
    warpAffine(templ, templ_warpped, rot_mat, rot_bbox.size());

    // check
    if ((imageROI.cols < templ_warpped.cols)
        || (imageROI.rows < templ_warpped.rows))
    {
      error_string = "36002| ROI is too small.";
      std::cout << error_string << " imageROI Size:" << imageROI.size()
                << ", templ_warpped Size:" << templ_warpped.size() << std::endl;
      return false;
    }

    result_cols = src.cols - templ_warpped.cols + 1;
    result_rows = src.rows - templ_warpped.rows + 1;
    result = Mat::zeros(result_cols, result_rows, CV_32FC1);

    // Mat mask =
    matchTemplate(imageROI, templ_warpped, result, match_method);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
    {
      if (best_value > minVal)
      {
        best_loc = imageROI_origin_in_src + minLoc;
        best_value = minVal;
        best_angle = angle;
      }
    }
    else
    {
      if (best_value < maxVal)
      {
        best_loc = imageROI_origin_in_src + maxLoc;
        best_value = maxVal;
        best_angle = angle;
      }
    }
  }

  return true;
}

void ModelDetector::pixResult2PhyResult(
  std::vector<ModelDetector::MatchResult>& match_results, cv::Size2f model_size)
{
  for (auto& result : match_results)
  {
    double best_angle = result.best_angle;
    cv::Point best_loc = result.best_loc;

    cv::Point2f model_center =
      Point2f(model_size.width / 2., model_size.height / 2.);
    // calculate the size of the finsal best bbox
    Mat M_max = getRotationMatrix2D(model_center, best_angle, 1.);
    cv::Rect max_bbox =
      cv::RotatedRect(model_center, model_size, best_angle).boundingRect();

    // change the center of rotation
    M_max.at<double>(0, 2) += max_bbox.width / 2.0 - model_center.x;
    M_max.at<double>(1, 2) += max_bbox.height / 2.0 - model_center.y;

    Mat axes = Mat::zeros(3, 1, CV_32FC2);
    axes.at<Vec2f>(0) = Vec2f(0, 0);
    axes.at<Vec2f>(1) = Vec2f(10, 0);
    axes.at<Vec2f>(2) = Vec2f(0, 10);

    Mat cur_axes;
    cv::transform(axes, cur_axes, M_max);  // cur_axes = axes*M_max
    Vec2f offset_maxloc = Vec2f(best_loc.x, best_loc.y);
    // Vec2f offset_roi = Vec2f(point_cloud2_range_.x, point_cloud2_range_.y);
    Vec2f offset_roi = Vec2f(min_corner_[0], min_corner_[1]);
    cur_axes.at<Vec2f>(0) =
      (cur_axes.at<Vec2f>(0) + offset_maxloc) * resolution_
      + offset_roi;  // add
    cur_axes.at<Vec2f>(1) =
      (cur_axes.at<Vec2f>(1) + offset_maxloc) * resolution_ + offset_roi;
    cur_axes.at<Vec2f>(2) =
      (cur_axes.at<Vec2f>(2) + offset_maxloc) * resolution_ + offset_roi;

    Vec2f axis_x = cur_axes.at<Vec2f>(1) - cur_axes.at<Vec2f>(0);
    double yaw = atan2(axis_x[1], axis_x[0]);

    Pose2D origin_pose;
    origin_pose.x = cur_axes.at<Vec2f>(0)[0];
    origin_pose.y = cur_axes.at<Vec2f>(0)[1];
    origin_pose.theta = yaw;
    result.cur_axes = cur_axes;
    result.model_in_map = origin_pose;

    cv::Mat four_corners = Mat::zeros(4, 1, CV_32FC2);
    Mat center = Mat::zeros(1, 1, CV_32FC2);
    Mat four_corners_trans;
    Mat center_trans;

    four_corners.at<Vec2f>(0) = Vec2f(25, 25);
    four_corners.at<Vec2f>(1) = Vec2f(model_size.width - 25, 25);
    four_corners.at<Vec2f>(2) =
      Vec2f(model_size.width - 25, model_size.height - 25);
    four_corners.at<Vec2f>(3) = Vec2f(25, model_size.height - 25);

    center.at<Vec2f>(0) = Vec2f(model_size.width / 2, model_size.height / 2);
    cv::transform(four_corners, four_corners_trans, M_max);
    cv::transform(center, center_trans, M_max);

    for (int i = 0; i < 4; i++)
    {
      Point2f corner =
        (four_corners_trans.at<Vec2f>(i) + offset_maxloc) * resolution_
        + offset_roi;
      std::cout << "four_corners[I]: " << i << ", " << corner << std::endl;
      result.corners[i] = corner;
    }

    result.center =
      (center_trans.at<Vec2f>(0) + offset_maxloc) * resolution_ + offset_roi;
	std::cout << "center: " << result.center << std::endl;
  }
}

void ModelDetector::drawMatch(cv::Mat& img_show, const cv::Mat& model,
                              std::vector<MatchResult>& match_results)
{
  if (img_show.channels() == 1)
  {
    Mat rgb_img;
    cvtColor(img_show, rgb_img, CV_GRAY2BGR);
    img_show = rgb_img;
  }

  for (auto cand_index = 0; cand_index < match_results.size(); cand_index++)
  {
    cv::Point best_loc;
    double best_angle;

    ModelDetector::MatchResult cand_result = match_results[cand_index];
    best_loc = cand_result.best_loc;
    best_angle = cand_result.best_angle;

    if (cand_result.invalid)
      continue;

    cv::Point2f center = Point2f(model.cols / 2, model.rows / 2);
    // calculate the size of the finsal best bbox
    Mat M_max = getRotationMatrix2D(center, best_angle, 1.);
    cv::Rect max_bbox =
      cv::RotatedRect(center, model.size(), best_angle).boundingRect();

    // change the center of rotation
    M_max.at<double>(0, 2) += max_bbox.width / 2.0 - center.x;
    M_max.at<double>(1, 2) += max_bbox.height / 2.0 - center.y;

    // show the template

    Mat templ_warpped;
    warpAffine(model, templ_warpped, M_max, max_bbox.size());  // change

    for (int i = 0; i < templ_warpped.rows; i++)
      for (int j = 0; j < templ_warpped.cols; j++)
      {
        int value = templ_warpped.at<uchar>(i, j);
        if ((best_loc.y + i) < img_show.rows
            && (best_loc.x + j) < img_show.cols)
        {
          if (value > 128)
            // map the template to roi scan
            img_show.at<Vec3b>(best_loc.y + i, best_loc.x + j) =
              Vec3b(0, value, 0);
        }
      }
    //    Mat cur_axes = cand_result.cur_axes;
    //    cv::line(img_show, cv::Point(cur_axes.at<Vec2f>(0)[0],
    //    cur_axes.at<Vec2f>(0)[1]),cv::Point(cur_axes.at<Vec2f>(1)[0],
    //    cur_axes.at<Vec2f>(1)[1]),CV_RGB(255,0,0));
    //    cv::line(img_show, cv::Point(cur_axes.at<Vec2f>(0)[0],
    //    cur_axes.at<Vec2f>(0)[1]),cv::Point(cur_axes.at<Vec2f>(2)[0],
    //    cur_axes.at<Vec2f>(2)[1]),CV_RGB(0,255,0));
  }
}

AwesomeClock::AwesomeClock()
{
}
AwesomeClock::~AwesomeClock()
{
}

void AwesomeClock::start()
{
  t_point_ = std::chrono::steady_clock::now();
}

double AwesomeClock::stop()
{
  time_elapse_ = std::chrono::steady_clock::now() - t_point_;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time_elapse_)
           .count()
         * 1.0e-9;
}