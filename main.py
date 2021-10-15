import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
from geopy.geocoders import Nominatim




# USER INPUT
#city_name = "Seattle"
# or try:
#city_name = "Chicago"
#city_name = "Philadelphia"
#city_name = "Brooklyn"
#city_name = "Miami"
#city_name = "Berkeley"





def do_all(city_name, minPts):
    geolocator = Nominatim(user_agent="name")
    location = geolocator.geocode(city_name)
    true_lat, true_long = location.latitude, location.longitude

    df = pd.read_csv('./data_files/gun-violence-data_01-2013_03-2018.csv')
    # set CooRdinate System for latitude / longitude
    my_crs = "EPSG:4326"

    my_map = gpd.read_file('./map_files/' + city_name + '_Streets.shp')

    # grab cities, lat, long
    city_df = df.loc[(df["city_or_county"] == city_name), ["latitude", "longitude"]]
    city_df.dropna(inplace=True)
    # Are these locations in the actual city. verify
    # whole coordinate point
    tol = .25
    city_df.drop(city_df.loc[(city_df['latitude'] > true_lat + tol) |
                             (city_df['latitude'] < true_lat - tol) |
                             (city_df['longitude'] > true_long + tol) |
                             (city_df['longitude'] < true_long - tol)].index, inplace=True)

    # list of points, long/ lat order matters
    city_geometry = geopandas.points_from_xy(city_df.longitude, city_df.latitude)
    # [Point(xy) for xy in zip( city_df["longitude"], city_df["latitude"])]

    geo_df = gpd.GeoDataFrame(city_df, crs=my_crs, geometry=city_geometry)

    fig, ax = plt.subplots(figsize=(40, 40))
    my_map.plot(ax=ax, alpha=0.4, color="grey")
    geo_df.plot(ax=ax, markersize=15, color="red", marker="o")

    plt.savefig("./outputs/" + city_name)

    # find most common location and count number of incdences
    most_common_locs = city_df.groupby(['latitude', 'longitude']).size().sort_values(ascending=False)
    # Seattle is not accuratly reporting locations of violent gun crimes
    # data say 88 shootings occured at sea-tac from 2013-2018!

    # find city with highest number of incidents
    cities_by_num = df.groupby('city_or_county').size().sort_values(ascending=False)
    # 1. Chicago: 11k,  2. Baltimore: 4k

    # START of DBSCAN
    # get array of lat and longs
    del city_df['geometry']
    xy_norm = city_df.to_numpy()
    # standardize
    sc = StandardScaler()
    xy_norm = sc.fit_transform(xy_norm)
    # get rid of nans
    xy_norm = xy_norm[~np.isnan(xy_norm).any(axis=1)]
    # get 4th farthest neighbor distance for each point,
    #   4 comes from oginal DBSCAN paper, 4 is best for all 2-d data (cite)
    nbrs = NearestNeighbors(n_neighbors=minPts + 1).fit(xy_norm)
    distances, indices = nbrs.kneighbors(xy_norm)



    dist_4th = np.delete(distances, list(range(0, minPts)), 1)
    # not sure why it doesnt want to sort
    # dist_4th_df = pd.DataFrame(dist_4th, columns=['4-dist'])
    # dist_4th_df.sort_values('4-dist', ascending=False)

    dist_4th_list = dist_4th.tolist()
    dist_4th_list.sort(reverse=True)
    dist_4th_arr = np.array(dist_4th_list)

    x = np.arange(0, len(dist_4th_list))
    # normalize dists
    minMaxSc = MinMaxScaler(feature_range=(0, 1))
    norm_dist = minMaxSc.fit_transform(dist_4th_arr)
    y = norm_dist[x]
    dist_str = str(minPts) + "-dist"
    # find the 'knee' automatically. critical eps value (https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf)
    norm_dist_df = pd.DataFrame(data=norm_dist, columns=[dist_str])
    min_neighbors = minPts
    norm_dist_pcts = norm_dist_df.pct_change()

    norm_dist_pcts_arr = norm_dist_pcts.to_numpy()

    # if change between points is less than 1% for minPts (4) times, then that is where eps should lie
    tol_minPts = minPts
    tol_per = -.001
    count = 0
    eps_norm = 0
    position = 0
    for j in np.nditer(norm_dist_pcts_arr):
        if j > tol_per:
            count = count + 1
        else:
            count = 0
        if count == tol_minPts:
            break
        position = position + 1
    if position == len(norm_dist):
        # nothing was found: minPts too large
        raise Exception("minPts: " + str(minPts) + " is too large for " + city_name)

    eps_norm = norm_dist[position]

    # get value back. undo [0, 1] scale
    eps = dist_4th_arr[position]

    plt.clf()
    plt.title("Radius Threshold:" + city_name)
    plt.xlabel("points")
    plt.ylabel(dist_str)
    plt.plot(x, y, "ob")
    plt.plot(position, eps_norm, marker=">", markersize=20, color="red")
    plt.savefig("./outputs/" + city_name + "RadiusGraph" + str(minPts))

    # find good radius from 4-dist without graph, so it can be automated
    radius = eps

    db = DBSCAN(eps=radius, min_samples=minPts).fit(xy_norm)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # number of clusters in labels
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # plot those clusters
    # drop nans
    geo_df.dropna(inplace=True)
    geo_df["label"] = labels.tolist()
    plt.clf()

    # get random colors
    n_colors = n_clusters_
    hexadecimal_alphabets = '0123456789ABCDEF'
    color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in
                            range(6)]) for i in range(n_colors)]

    # evaluation
    sil_score = metrics.silhouette_score(xy_norm, labels, metric='euclidean')
    sil_score_form = "{:.2f}".format(sil_score)
    sil_score_str = str(sil_score_form)

    city_geometry = geopandas.points_from_xy(geo_df.longitude, geo_df.latitude)

    geo_df = gpd.GeoDataFrame(geo_df, crs=my_crs, geometry=city_geometry)

    fig1, ax1 = plt.subplots(figsize=(40, 40))
    my_map.plot(ax=ax1, alpha=0.4, color="grey")
    geo_df[geo_df['label'] == -1].plot(ax=ax1, markersize=30, color="black", marker="o", label="outlier")
    for n in range(n_colors):
        geo_df[geo_df['label'] == n].plot(ax=ax1, markersize=30, color=color[n], marker="o", label=n + 1)
    ax1.legend(prop={'size': 15}, title=city_name + " Silhouette Score: " + sil_score_str + ", minPts: " + str(minPts))

    plt.savefig("./outputs/" + city_name + "Results_DBSCAN" + str(minPts))

    print()

#do_all("Seattle", 4) # 712 cases
#do_all("Seattle", 8)

#do_all("Philadelphia", 12)


do_all("Chicago", 525) # 11k cases
#do_all("Philadelphia") # 3k cases
#do_all("Brooklyn", 30) # 1.5 cases
#o_all("Miami") #

