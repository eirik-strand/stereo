import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import pptk


sensor_size = 0.036
focal_length = 0.05
camera_angle = math.atan2(sensor_size, (2.0*focal_length))*2.0
cam1_x = 0
cam2_x = 1

cam2_angle = np.radians(20.0)

l = 5


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


class Image_Stitching():
    def __init__(self):
        self.ratio = 0.5
        self.min_match = 10
        self.sift = cv2.xfeatures2d.SIFT_create()

    def registration(self, img1, img2, img3):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        good_points = []
        good_matches = []

        for m1, m2 in raw_matches:
            #print("m1", m1.trainIdx, m1.queryIdx)
            #print("m2", m2.trainIdx, m2.queryIdx)
            #print("m1 dfdfd", m1.imgIdx, m1.imgIdx)
            #print("m2 dfdfd", m2.imgIdx, m2.imgIdx)

            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('results/matching.jpg', img3)

        image1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])

        x_ = []
        y_ = []

        num_cols = img1.shape[1]
        num_rows = img2.shape[0]
        points = []
        for point1, point2 in zip(image1_kp, image2_kp):

            c_x = (img2.shape[1]/2.0)/(math.tan(camera_angle/2.0))
            angle1 = math.atan((point1[0]-img2.shape[1]/2.0)/c_x)
            angle2 = math.atan((point2[0]-img2.shape[1]/2.0)/c_x)
            #angle1 = angle_per_pix*((point1[0]-img2.shape[1]/2)*(math.cos(angle1/1.3)))
            #angle2 = angle_per_pix*((point2[0]-img2.shape[1]/2)*(math.cos(angle2/1.3)))

            line1 = [[cam1_x, 0],
                     [cam1_x + (l)*math.tan(angle1), l]]
            line2 = [[cam2_x, 0], [
                cam2_x + (l)*math.tan(angle2-cam2_angle), l]]

            intersect = line_intersection(line1, line2)
            plt.scatter(intersect[0], intersect[1], 3,
                        linewidths=0.0, color="red")
            # plt.plot(line1, color='red')
            # plt.plot(line2, color='green')
            points.append([intersect[0], intersect[1], 0])

        # v = pptk.viewer(points)
        # v.set(point_size=0.05)
        return []

    def getComponents(self, normalised_homography):
        '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
        a = normalised_homography[0, 0]
        b = normalised_homography[0, 1]
        c = normalised_homography[0, 2]
        d = normalised_homography[1, 0]
        e = normalised_homography[1, 1]
        f = normalised_homography[1, 2]

        p = math.sqrt(a*a + b*b)
        r = (a*e - b*d)/(p)
        q = (a*d+b*e)/(a*e - b*d)

        translation = (c, f)
        scale = (p, r)
        shear = q
        theta = math.atan2(b, a)

        return (translation, theta, scale, shear)

    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2][1], R[2][2])
            y = math.atan2(-R[2][0], sy)
            z = math.atan2(R[1][0], R[0][0])
        else:
            x = math.atan2(-R[1][2], R[1][1])
            y = math.atan2(-R[2][0], sy)
            z = 0
        return (np.rad2deg(x), np.rad2deg(y), np.rad2deg(z))

    def blending(self, img1, img2, img3):
        self.registration(img1, img2, img3)

        return img1


if __name__ == '__main__':
    img2 = cv2.imread('data/stereoR/concrete.jpg')
    global calculated_angle
    img1 = cv2.imread('data/stereoL/concrete.jpg')
    img3 = cv2.imread('data/stereoL/concrete.jpg')
    final = Image_Stitching().blending(img1, img2, img3)
    cv2.imwrite('results/stitch.jpg', final)

    print(img2.shape[1])

    plt.plot([cam1_x, cam1_x-l*math.tan(camera_angle/2)],
             [0, l], color='black')
    plt.plot([cam1_x, cam1_x+l*math.tan(camera_angle/2)],
             [0, l], color='black')

    plt.plot([cam2_x, cam2_x -
              l*math.tan(camera_angle/2+cam2_angle)], [0, l], color='black')
    plt.plot([cam2_x, cam2_x +
              l*math.tan(camera_angle/2-cam2_angle)], [0, l], color='black')

    plt.xlim(-1, 1.4)
    plt.ylim(0, 4)

    plt.grid(linestyle='dotted')

    plt.savefig('cross.eps', format='eps')
