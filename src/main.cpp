#include "model_detector.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cout << "usage: ./model_detect_test src_img  templ_img";
    return 0;
  }
  Mat src = imread(argv[1], 0);
  Mat templ = imread(argv[2], 0);
  string ee;
  ModelDetector md;
  std::vector<ModelDetector::MatchResult> match_results =
    md.pyramidMatching(src, templ, ee, 0.9, 3);

  Mat src_paint;
  cvtColor(src, src_paint, CV_GRAY2BGR);
  md.drawMatch(src_paint, templ, match_results);

  imwrite("result.jpg", src_paint);
  imshow("result.jpg", src_paint);
  waitKey(0);

  return 0;
}
