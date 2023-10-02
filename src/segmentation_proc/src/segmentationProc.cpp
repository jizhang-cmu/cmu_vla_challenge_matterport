#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;

const double PI = 3.1415926;

string seg_file_dir;
int segDisplayInterval = 2;
int segDisplayCount = 0;

// define region point
struct RegionPoint {
     float x, y, z;
     float box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;
     float height;
     int region_index, level_index;
     std::uint8_t label[1024];
};

POINT_CLOUD_REGISTER_POINT_STRUCT (RegionPoint,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, box_min_x, box_min_x)
                                   (float, box_min_y, box_min_y)
                                   (float, box_min_z, box_min_z)
                                   (float, box_max_x, box_max_x)
                                   (float, box_max_y, box_max_y)
                                   (float, box_max_z, box_max_z)
                                   (float, height, height)
                                   (int, region_index, region_index)
                                   (int, level_index, level_index)
                                   (std::uint8_t[1024], label, label))

// define object point
struct ObjectPoint {
     float x, y, z;
     float axis0_x, axis0_y, axis0_z;
     float axis1_x, axis1_y, axis1_z;
     float radius_x, radius_y, radius_z;
     int object_index, region_index, category_index;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (ObjectPoint,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, axis0_x, axis0_x)
                                   (float, axis0_y, axis0_y)
                                   (float, axis0_z, axis0_z)
                                   (float, axis1_x, axis1_x)
                                   (float, axis1_y, axis1_y)
                                   (float, axis1_z, axis1_z)
                                   (float, radius_x, radius_x)
                                   (float, radius_y, radius_y)
                                   (float, radius_z, radius_z)
                                   (int, object_index, object_index)
                                   (int, region_index, region_index)
                                   (int, category_index, category_index))

pcl::PointCloud<RegionPoint>::Ptr regionAll(new pcl::PointCloud<RegionPoint>());
pcl::PointCloud<ObjectPoint>::Ptr objectAll(new pcl::PointCloud<ObjectPoint>());

double systemTime = 0;

float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;

sensor_msgs::PointCloud2 regionAll2, objectAll2;
visualization_msgs::MarkerArray regionMarkerArray, objectMarkerArray;

int readSegmentationFile(const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open house file %s\n", filename);
    return 0;
  }

  char cmd[1024], version[1024], name_buffer[1024], label_buffer[1024];
  int nimages, npanoramas, nvertices, nsurfaces, nsegments, nobjects, ncategories, nregions, nportals, nlevels, val;
  int house_index, level_index, region_index, surface_index, category_index, object_index, panorama_index, id, dummy;
  double height, area;
  double position[3];
  double normal[3];
  double box[2][3];

  // read file type and version
  val = fscanf(fp, "%s%s", cmd, version);
  if (strcmp(cmd, "ASCII")) {
    fprintf(stderr, "Unable to read ascii file %s, wrong type: %s\n", filename, cmd);
    return 0;
  }

  // read header
  if (!strcmp(version, "1.0")) {
    nsegments = 0;
    nobjects = 0;
    ncategories = 0;
    nportals = 0;
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%s", name_buffer);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%d", &nimages);
    val = fscanf(fp, "%d", &npanoramas);
    val = fscanf(fp, "%d", &nvertices);
    val = fscanf(fp, "%d", &nsurfaces);
    val = fscanf(fp, "%d", &nregions);
    val = fscanf(fp, "%d", &nlevels);
    val = fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int i = 0; i < 8; i++) { val = fscanf(fp, "%d", &dummy); }
  }
  else {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%s", name_buffer);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%d", &nimages);
    val = fscanf(fp, "%d", &npanoramas);
    val = fscanf(fp, "%d", &nvertices);
    val = fscanf(fp, "%d", &nsurfaces);
    val = fscanf(fp, "%d", &nsegments);
    val = fscanf(fp, "%d", &nobjects);
    val = fscanf(fp, "%d", &ncategories);
    val = fscanf(fp, "%d", &nregions);
    val = fscanf(fp, "%d", &nportals);
    val = fscanf(fp, "%d", &nlevels);
    for (int i = 0; i < 5; i++) { val = fscanf(fp, "%d", &dummy); }
    val = fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int i = 0; i < 5; i++) { val = fscanf(fp, "%d", &dummy); }
  }

  // read levels
  for (int i = 0; i < nlevels; i++) {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &dummy);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    val = fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int j = 0; j < 5; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "L")) { fprintf(stderr, "Error reading level %d\n", i); return 0; }
  }
    
  // read regions
  for (int i = 0; i < nregions; i++) {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &level_index);
    val = fscanf(fp, "%d%d", &dummy, &dummy);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    val = fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    val = fscanf(fp, "%lf", &height);
    for (int j = 0; j < 4; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "R")) { fprintf(stderr, "Error reading region %d\n", i); return 0; }
    RegionPoint regPoint;
    regPoint.x = position[0];
    regPoint.y = position[1];
    regPoint.z = position[2];
    regPoint.box_min_x = box[0][0];
    regPoint.box_min_y = box[0][1];
    regPoint.box_min_z = box[0][2];
    regPoint.box_max_x = box[1][0];
    regPoint.box_max_y = box[1][1];
    regPoint.box_max_z = box[1][2];
    regPoint.height = height;
    regPoint.region_index = house_index;
    regPoint.level_index = level_index;
    strncpy((char*)regPoint.label, label_buffer, 1024);
    regionAll->points.push_back(regPoint);
  }
    
  // read portals
  for (int i = 0; i < nportals; i++) {
    int region0_index, region1_index;
    double p0[3], p1[3];
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &region0_index);
    val = fscanf(fp, "%d", &region1_index);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%lf%lf%lf", &p0[0], &p0[1], &p0[2]);
    val = fscanf(fp, "%lf%lf%lf", &p1[0], &p1[1], &p1[2]);
    for (int j = 0; j < 4; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "P")) { fprintf(stderr, "Error reading portal %d\n", i); return 0; }
  }
    
  // read surfaces
  for (int i = 0; i < nsurfaces; i++) {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &region_index);
    val = fscanf(fp, "%d", &dummy);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    val = fscanf(fp, "%lf%lf%lf", &normal[0], &normal[1], &normal[2]);
    val = fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int j = 0; j < 5; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "S")) { fprintf(stderr, "Error reading surface %d\n", i); return 0; }
  }
    
  // read vertices
  for (int i = 0; i < nvertices; i++) {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &surface_index);
    val = fscanf(fp, "%s", label_buffer);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    val = fscanf(fp, "%lf%lf%lf", &normal[0], &normal[1], &normal[2]);
    for (int j = 0; j < 3; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "V")) { fprintf(stderr, "Error reading vertex %d\n", i); return 0; }
  }

  // read panoramas
  for (int i = 0; i < npanoramas; i++) {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%s", name_buffer);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &region_index);
    val = fscanf(fp, "%d", &dummy);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    for (int j = 0; j < 5; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "P")) { fprintf(stderr, "Error reading panorama %d\n", i); return 0; }
  }

  // read images
  for (int i = 0; i < nimages; i++) {
    double intrinsics[9];
    double extrinsics[16];
    int camera_index, yaw_index, width, height;
    char depth_filename[1024], color_filename[1024];
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &panorama_index);
    val = fscanf(fp, "%s", name_buffer);
    val = fscanf(fp, "%d", &camera_index);
    val = fscanf(fp, "%d", &yaw_index);
    for (int j = 0; j < 16; j++) { val = fscanf(fp, "%lf", &extrinsics[j]); }
    for (int j = 0; j < 9; j++) { val = fscanf(fp, "%lf", &intrinsics[j]); }
    val = fscanf(fp, "%d%d", &width, &height);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    for (int j = 0; j < 5; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "I")) { fprintf(stderr, "Error reading image %d\n", i); return 0; }
  }

  // read categories
  for (int i = 0; i < ncategories; i++) {
    int label_id, mpcat40_id;
    char label_name[1024], mpcat40_name[1024];
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d %s", &label_id, label_name);
    val = fscanf(fp, "%d %s", &mpcat40_id, mpcat40_name);
    for (int j = 0; j < 5; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "C")) { fprintf(stderr, "Error reading category %d\n", i); return 0; }
  }
    
  // read objects
  for (int i = 0; i < nobjects; i++) {
    double axis0[3], axis1[3], radius[3];
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &region_index);
    val = fscanf(fp, "%d", &category_index);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    val = fscanf(fp, "%lf%lf%lf", &axis0[0], &axis0[1], &axis0[2]);
    val = fscanf(fp, "%lf%lf%lf", &axis1[0], &axis1[1], &axis1[2]);
    val = fscanf(fp, "%lf%lf%lf", &radius[0], &radius[1], &radius[2]);
    for (int j = 0; j < 8; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "O")) { fprintf(stderr, "Error reading object %d\n", i); return 0; }
    ObjectPoint objPoint;
    objPoint.x = position[0];
    objPoint.y = position[1];
    objPoint.z = position[2];
    objPoint.axis0_x = axis0[0];
    objPoint.axis0_y = axis0[1];
    objPoint.axis0_z = axis0[2];
    objPoint.axis1_x = axis1[0];
    objPoint.axis1_y = axis1[1];
    objPoint.axis1_z = axis1[2];
    objPoint.radius_x = radius[0];
    objPoint.radius_y = radius[1];
    objPoint.radius_z = radius[2];
    objPoint.object_index = house_index;
    objPoint.region_index = region_index;
    objPoint.category_index = category_index;
    objectAll->points.push_back(objPoint);
  }
    
  // read segments
  for (int i = 0; i < nsegments; i++) {
    val = fscanf(fp, "%s", cmd);
    val = fscanf(fp, "%d", &house_index);
    val = fscanf(fp, "%d", &object_index);
    val = fscanf(fp, "%d", &id);
    val = fscanf(fp, "%lf", &area);
    val = fscanf(fp, "%lf%lf%lf", &position[0], &position[1], &position[2]);
    val = fscanf(fp, "%lf%lf%lf%lf%lf%lf", &box[0][0], &box[0][1], &box[0][2], &box[1][0], &box[1][1], &box[1][2]);
    for (int j = 0; j < 5; j++) { val = fscanf(fp, "%d", &dummy); }
    if (strcmp(cmd, "E")) { fprintf(stderr, "Error reading segment %d\n", i); return 0; }
  }
    
  fclose(fp);

  return 1;
}

void odometryHandler(const nav_msgs::Odometry::ConstPtr& odom)
{
  systemTime = odom->header.stamp.toSec();

  double roll, pitch, yaw;
  geometry_msgs::Quaternion geoQuat = odom->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  vehicleX = odom->pose.pose.position.x;
  vehicleY = odom->pose.pose.position.y;
  vehicleZ = odom->pose.pose.position.z;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "segmentationProc");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");

  nhPrivate.getParam("seg_file_dir", seg_file_dir);
  nhPrivate.getParam("segDisplayInterval", segDisplayInterval);

  ros::Subscriber subOdometry = nh.subscribe<nav_msgs::Odometry> ("/state_estimation", 5, odometryHandler);

  ros::Publisher pubRegion = nh.advertise<sensor_msgs::PointCloud2> ("/region_segmentations", 5);

  ros::Publisher pubObject = nh.advertise<sensor_msgs::PointCloud2> ("/object_segmentations", 5);

  ros::Publisher pubRegionMarker = nh.advertise<visualization_msgs::MarkerArray>("region_segmentation_markers", 10);

  ros::Publisher pubObjectMarker = nh.advertise<visualization_msgs::MarkerArray>("object_segmentation_markers", 10);

  // read segmentation file
  if (readSegmentationFile(seg_file_dir.c_str()) != 1) {
    printf("\nCouldn't read segmentation file.\n");
  }

  // prepare point cloud messages
  pcl::toROSMsg(*regionAll, regionAll2);
  pcl::toROSMsg(*objectAll, objectAll2);

  // prepare marker array messages
  int regionNum = regionAll->points.size();
  visualization_msgs::MarkerArray regionMarkerArray;
  regionMarkerArray.markers.resize(regionNum);
  for (int i = 0; i < regionNum; i++) {
    regionMarkerArray.markers[i].header.frame_id = "map";
    regionMarkerArray.markers[i].header.stamp = ros::Time().fromSec(systemTime);
    regionMarkerArray.markers[i].ns = "region";
    regionMarkerArray.markers[i].id = regionAll->points[i].region_index;
    regionMarkerArray.markers[i].action = visualization_msgs::Marker::ADD;
    regionMarkerArray.markers[i].type = visualization_msgs::Marker::CUBE;
    regionMarkerArray.markers[i].pose.position.x = (regionAll->points[i].box_min_x + regionAll->points[i].box_max_x) / 2.0;
    regionMarkerArray.markers[i].pose.position.y = (regionAll->points[i].box_min_y + regionAll->points[i].box_max_y) / 2.0;
    regionMarkerArray.markers[i].pose.position.z = (regionAll->points[i].box_min_z + regionAll->points[i].box_max_z) / 2.0;
    regionMarkerArray.markers[i].pose.orientation.x = 0.0;
    regionMarkerArray.markers[i].pose.orientation.y = 0.0;
    regionMarkerArray.markers[i].pose.orientation.z = 0.0;
    regionMarkerArray.markers[i].pose.orientation.w = 1.0;
    regionMarkerArray.markers[i].scale.x = regionAll->points[i].box_max_x - regionAll->points[i].box_min_x;
    regionMarkerArray.markers[i].scale.y = regionAll->points[i].box_max_y - regionAll->points[i].box_min_y;
    regionMarkerArray.markers[i].scale.z = regionAll->points[i].box_max_z - regionAll->points[i].box_min_z;
    regionMarkerArray.markers[i].color.a = 0.5;
    regionMarkerArray.markers[i].color.r = 1.0;
    regionMarkerArray.markers[i].color.g = 0;
    regionMarkerArray.markers[i].color.b = 0;
  }

  int objectNum = objectAll->points.size();
  visualization_msgs::MarkerArray objectMarkerArray;
  objectMarkerArray.markers.resize(objectNum);
  for (int i = 0; i < objectNum; i++) {
    objectMarkerArray.markers[i].header.frame_id = "map";
    objectMarkerArray.markers[i].header.stamp = ros::Time().fromSec(systemTime);
    objectMarkerArray.markers[i].ns = "object";
    objectMarkerArray.markers[i].id = objectAll->points[i].object_index;
    objectMarkerArray.markers[i].action = visualization_msgs::Marker::ADD;
    objectMarkerArray.markers[i].type = visualization_msgs::Marker::CUBE;
    objectMarkerArray.markers[i].pose.position.x = objectAll->points[i].x;
    objectMarkerArray.markers[i].pose.position.y = objectAll->points[i].y;
    objectMarkerArray.markers[i].pose.position.z = objectAll->points[i].z;
    objectMarkerArray.markers[i].pose.orientation = tf::createQuaternionMsgFromRollPitchYaw
                                                    (0, 0, atan2(objectAll->points[i].axis0_y, objectAll->points[i].axis0_x));
    objectMarkerArray.markers[i].scale.x = 2.0 * objectAll->points[i].radius_x;
    objectMarkerArray.markers[i].scale.y = 2.0 * objectAll->points[i].radius_y;
    objectMarkerArray.markers[i].scale.z = 2.0 * objectAll->points[i].radius_z;
    objectMarkerArray.markers[i].color.a = 0.1;
    objectMarkerArray.markers[i].color.r = 0;
    objectMarkerArray.markers[i].color.g = 1.0;
    objectMarkerArray.markers[i].color.b = 0;
  }

  printf("\nRead %d regions and %d objects.\n\n", regionNum, objectNum);

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();

    segDisplayCount++;
    if (segDisplayCount >= 100 * segDisplayInterval) {
      // publish point clouds with regions
      regionAll2.header.stamp = ros::Time().fromSec(systemTime);
      regionAll2.header.frame_id = "map";
      pubRegion.publish(regionAll2);

      // publish point clouds with objects
      objectAll2.header.stamp = ros::Time().fromSec(systemTime);
      objectAll2.header.frame_id = "map";
      pubObject.publish(objectAll2);

      // publish marker arrays
      pubRegionMarker.publish(regionMarkerArray);
      pubObjectMarker.publish(objectMarkerArray);

      segDisplayCount = 0;
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
