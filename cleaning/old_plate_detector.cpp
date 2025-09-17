FrameProcessor::FrameProcessor() {
    // get robot config info
    m_config_robot_yaml = Hive::env->robot_info->get_config();

    white_lower[0] = m_config_cv_yaml["white_lower"][0].as<int>();
    white_lower[1] = m_config_cv_yaml["white_lower"][1].as<int>();
    white_lower[2] = m_config_cv_yaml["white_lower"][2].as<int>();

    white_upper[0] = m_config_cv_yaml["white_upper"][0].as<int>();
    white_upper[1] = m_config_cv_yaml["white_upper"][1].as<int>();
    white_upper[2] = m_config_cv_yaml["white_upper"][2].as<int>();

    blue_lower[0] = m_config_cv_yaml["blue_lower"][0].as<int>();
    blue_lower[1] = m_config_cv_yaml["blue_lower"][1].as<int>();
    blue_lower[2] = m_config_cv_yaml["blue_lower"][2].as<int>();

    blue_upper[0] = m_config_cv_yaml["blue_upper"][0].as<int>();
    blue_upper[1] = m_config_cv_yaml["blue_upper"][1].as<int>();
    blue_upper[2] = m_config_cv_yaml["blue_upper"][2].as<int>();

    red_lower[0] = m_config_cv_yaml["red_lower"][0].as<int>();
    red_lower[1] = m_config_cv_yaml["red_lower"][1].as<int>();
    red_lower[2] = m_config_cv_yaml["red_lower"][2].as<int>();

    red_upper[0] = m_config_cv_yaml["red_upper"][0].as<int>();
    red_upper[1] = m_config_cv_yaml["red_upper"][1].as<int>();
    red_upper[2] = m_config_cv_yaml["red_upper"][2].as<int>();
    

    


FrameProcessor::FrameDetections FrameProcessor::detect_armor_plates_realsense(cv::Mat color_image, rs2::video_frame color_frame, rs2::depth_frame depth_frame, std::string color, bool debug, bool show_detection, double time_stamp) {

    FrameProcessor::FrameDetections frame_detections;
    // if(i < 1000){
    //     m_video_writer.write(color_image);
    //     i++;
    // }
    // if(i == 1000) {
    //     m_video_writer.release();
    // }

    if (debug) {
        Hive::env->waggle->display_cv_mat("Input Frame", color_image);
    }

    ///////////////////////////////
    /// Things to get from YAML ///
    ///////////////////////////////

    int min_pixel_area = 5;
    float min_h_to_w_ratio = 1.0;
    float min_light_overlap = 0.8;
    float min_light_size_ratio = 0.1;
    float min_width_to_height_ratio = .8; // 1
    float max_width_to_height_ratio = 6;
    float min_inner_width_respect_to_light_width = 1000;//1.5;

    //Min heght to search for symbol
    // int inner_symbol_search_height = 3;
    //How big the symbol can be in relation to target
    float min_symbol_to_target_height_ratio = 0.6;
    // Max offset x symbol can be from center of target
    float max_symbol_x_offset_from_center = 0.15;
    // Max offset y symbol can be from center of target
    float max_symbol_y_offset_from_center = 3; // 0.3

    /////////////////////
    /// END OF THINGS ///
    /////////////////////


    /////////////////////////////////
    /// Create Masks and Contours ///
    /////////////////////////////////

    cv::Mat debug_image(color_image.size(), 0);
    cv::Mat hsv_image(color_image.size(), 0);
    cv::Mat mask_white(color_image.size(), 0);
    cv::Mat mask_color(color_image.size(), 0);
    color_image.copyTo(debug_image);

    cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, white_lower, white_upper, mask_white);

    // Threshold color_mask based on target teams color
    bool is_blue = color == "blue";
    if (is_blue) {
        cv::inRange(hsv_image, blue_lower, blue_upper, mask_color);
    } else if (color == "red") {
        // We need to ranges for red masks since it is on both ends of the hsv spectrum
        cv::Mat red_mask1, red_mask2;

        cv::Scalar red_lower_1(red_lower[0] + 180, red_lower[1], red_lower[2]);
        cv::Scalar red_upper_1(180, red_upper[1], red_upper[2]);
        cv::Scalar red_lower_2(0, red_lower[1], red_lower[2]);
        cv::Scalar red_upper_2 = red_upper;

        cv::inRange(hsv_image, red_lower_1, red_upper_1, red_mask1);
        cv::inRange(hsv_image, red_lower_2, red_upper_2, red_mask2);
        if (debug) {
            Hive::env->waggle->display_cv_mat("Red Mask 1", red_mask1);
            Hive::env->waggle->display_cv_mat("Red Mask 2", red_mask2);
        }
        cv::bitwise_or(red_mask1, red_mask2, mask_color);
    } else {
        std::cerr << "ERROR: Invalid target color '" + color + "'" << std::endl;
        return frame_detections;
    }
    cv::dilate(mask_color, mask_color, cv::Mat(), cv::Point(-1, -1), 3);
    cv::erode(mask_color, mask_color, cv::Mat(), cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> filtered_light_contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Rect> filtered_lights;
    cv::findContours(mask_color, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat mask_color_blocked(color_image.size(), CV_8UC1, cv::Scalar(0));

    //creates a rectangle around all of the found contours from the team color mask
    for (std::vector<cv::Point> cnt : contours) {
        cv::Rect rect = cv::boundingRect(cnt);
        // Filters out select contours based on their area, and their height to width ratio
        int w = rect.width, h = rect.height;
        if ((w * h) < min_pixel_area) continue;
        if (h / w < min_h_to_w_ratio) continue;
        cv::rectangle(mask_color_blocked, rect, 255, -1);
        filtered_lights.push_back(rect);
        filtered_light_contours.push_back(cnt);
        if (debug) {
            cv::rectangle(debug_image, rect, { 200, 104, 52 }, -1);
        }
    }

    //////////////////////////
    // Filter Armor Plates ///
    //////////////////////////

    // Think of each pair of countours could be considered as a potential armor palte
    // The following Filters are based on this idea
    float over_lap_multiplier = (min_light_overlap / 2.0) + 5;
    for (size_t i = 0; i < filtered_lights.size(); i++) {
        std::vector<cv::Point> light1_cnt = filtered_light_contours[i];
        cv::Rect light1 = filtered_lights[i];
        float light1_overlap_height = float(light1.y) + (float(light1.height) * over_lap_multiplier);

        for (size_t j = i + 1; j < filtered_lights.size(); j++) {
            cv::Rect light2 = filtered_lights[j];

            std::vector<cv::Point> light2_cnt = filtered_light_contours[j];

            float light2_overlap_height = float(light2.y) + (float(light2.height) * over_lap_multiplier);

            // Filters based on the vertical overlap between the two lights
            // You can image horizontal lines drawn from the top and bottom of both lights
            // This is filtering based on how much those regions overlap
            if (light1_overlap_height < light2.y || light2_overlap_height < light1.y) continue;

            float vertical_size_ratio = float(light1.height) / float(light2.height);

            // Filters based on the ratio between the height of the two potential ligthts
            if (vertical_size_ratio < min_light_size_ratio || vertical_size_ratio >(1.0 / min_light_size_ratio)) continue;

            // Creates a rectangle of the potential armor plate
            int target_x = std::min(light1.x, light2.x);
            int target_y = std::min(light1.y, light2.y);
            int target_w = std::max(light1.x + light1.width, light2.x + light2.width) - target_x;
            int target_h = std::max(light1.y + light1.height, light2.y + light2.height) - target_y;
            cv::Rect target_rect(target_x, target_y, target_w, target_h);
            // Filters based on the width to height ration of the potential armor plate
            if (((float(target_w) / float(target_h)) < min_width_to_height_ratio) || ((float(target_w) / float(target_h)) > max_width_to_height_ratio)) continue;

            /////////////////////////////////////////
            /// Filters Based on the Inner Target ///
            /////////////////////////////////////////

            // the area of the target without the lights and including the number
            int inner_x = std::min((light1.x + light1.width), (light2.x + light2.width));
            int inner_y = target_y;
            int inner_w = std::max(light1.x, light2.x) - inner_x;
            int inner_h = target_h;

            // Filters the inner target width to light width ratio for both lights
            if ((inner_w * min_inner_width_respect_to_light_width < light1.width)
                || (inner_w * min_inner_width_respect_to_light_width < light2.width)) continue;

            if (debug) cv::rectangle(debug_image, { inner_x, inner_y }, { inner_x + inner_w, inner_y + inner_h }, { 100, 100, 0 }, 3);
            cv::Rect inner_target(inner_x, inner_y, inner_w, inner_h);

            cv::Mat mask_color_target = mask_color_blocked(inner_target);

            float inner_color_density = cv::sum(mask_color_target)[0]; //index 0 should be right since the the image is grayscale

            // Filters to ensure the lights are not present inside the inner target

            if (inner_color_density > 0) continue;

            

/// @brief converts a realsense color frame into a an open cv Mat
/// @param frame the realsense video frame
/// @return the video frame as  an open cv::Mat
cv::Mat FrameProcessor::frame_to_mat(const rs2::frame& frame) {
    using namespace cv;
    using namespace rs2;

    auto vf = frame.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (frame.get_profile().format() == RS2_FORMAT_BGR8) {
        return Mat(Size(w, h), CV_8UC3, (void*)frame.get_data(), Mat::AUTO_STEP);
    } else if (frame.get_profile().format() == RS2_FORMAT_RGB8) {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void*)frame.get_data(), Mat::AUTO_STEP);
        cv::Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    } else if (frame.get_profile().format() == RS2_FORMAT_Z16) {
        return Mat(Size(w, h), CV_16UC1, (void*)frame.get_data(), Mat::AUTO_STEP);
    } else if (frame.get_profile().format() == RS2_FORMAT_Y8) {
        return Mat(Size(w, h), CV_8UC1, (void*)frame.get_data(), Mat::AUTO_STEP);

    } else if (frame.get_profile().format() == RS2_FORMAT_DISPARITY32) {
        return Mat(Size(w, h), CV_32FC1, (void*)frame.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");


}