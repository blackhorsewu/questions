# find feature value of each point of the point cloud and put them into a list
def find_feature_value(pcd):

  # Build a KD (k-dimensional) Tree for Flann
  # Fast Library for Approximate Nearest Neighbor
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)

  # Treat pcd.points as an numpy array of n points by m attributes of a point
  # The first dimension, shape[0], of this array is the number of points in this point cloud.
  pc_number = np.asarray(pcd.points).shape[0]

  feature_value_list = []
  
  # This is very important. It specifies the attribute that we are using to find the feature
  # when it is pcd.normals, it is using the normals to find the feature,
  # n_list is an array of normals of all the points
  n_list = np.asarray(pcd.normals)

  # a // b = integer quotient of a divided by b
  # so neighbor (number of neighbors) whichever is smaller of 30 or the quotient 
  # of dividing the number of points by 100
  # neighbour = min(pc_number//100, 30)
  neighbour = feature_neighbours
  print("Feature value neighbour: ", neighbour)
  # for every point of the point cloud
  for index in range(pc_number):
      
      # Search the k nearest neighbour for each point of the point cloud.
      # The pcd.points[index] is the (query) point to find the nearest neighbour for.
      # 'neighbour', found above, is the number of neighbours to be searched.
      [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], neighbour)

      # get rid of the query point in the neighbourhood
      idx = idx[1:]

      # n_list, computed above, is an array of normals of every point.
      # 'vector' is then a vector with its components the arithmetic mean of every
      # element of all the k neighbours of that (query) point
      # This can be called the CENTROID of the NORMALs of its neighbours
      vector = np.mean(n_list[idx, :], axis=0)
      
      # the bigger the feature value, meaning the normal of that point is more 
      # different from its neighbours
      feature_value = np.linalg.norm(
          vector - (n_list[index, :] * np.dot(vector, n_list[index, :])))
      feature_value_list.append(feature_value)

  return np.array(feature_value_list)
