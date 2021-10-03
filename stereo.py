import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

camera_angle = np.radians(40)
cam1_x = 0
cam2_x = 0.455
sensor_size = 0.035
cam2_angle = np.radians(20.0)

l = 2


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
        self.ratio = 0.6
        self.min_match = 10
        self.sift = cv2.xfeatures2d.SIFT_create()

    def registration(self, img1, img2, img3):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        kp3, des3 = self.sift.detectAndCompute(img3, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        raw_matches2 = matcher.knnMatch(des3, des1, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)

        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(
                image2_kp, image1_kp, cv2.RANSAC, 5.0)
        # cv2.normalize(H, H)

        x_ = []
        y_ = []

        num_cols = img1.shape[1]
        num_rows = img2.shape[0]

        for point1, point2 in zip(image1_kp, image2_kp):
            x_.append(point1[0]-point2[0])
            y_.append(point1[1]-point2[1])

            ratio1 = point1[0]/num_cols
            ratio2 = point2[0]/num_cols
            v_cam_angle = camera_angle*num_rows/num_cols
            v_angle1 = v_cam_angle*((point1[1])/num_rows-0.5)
            v_angle2 = v_cam_angle*((point2[1])/num_rows-0.5)
            r1 = math.cos(v_angle1)
            r2 = math.cos(v_angle2)
            angle1 = camera_angle*((point1[0])/num_cols-0.5)
            angle2 = camera_angle*((point2[0])/num_cols-0.5)

            line1 = [[0, 0], [1, 1]]
            line2 = [[0, 0], [1, 2]]

            line1 = [[cam1_x, 0],
                     [cam1_x + (l)*math.tan(angle1), l]]
            line2 = [[cam2_x, 0], [
                cam2_x + (l)*math.tan(angle2-cam2_angle), l]]

            intersect = line_intersection(line1, line2)

            plt.scatter(intersect[0], intersect[1], 0.5, linewidths="0.0")
            # plt.plot(line1, color='red')
            # plt.plot(line2, color='green')
        x_median = np.median(x_)
        y_median = np.median(y_)

        global calculated_angle
        global match_ration
        match_ration = x_median/num_cols
        calculated_angle = camera_angle*x_median/num_cols
        print(match_ration)
        print(calculated_angle)
        return H

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
        H = self.registration(img1, img2, img3)
        print(H)

        K = np.array([[2666.7, 0, 960.0], [0, 2666.7, 540.0], [0, 0, 1]]
                     )

        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(
            H, K)

        for R in Rs:

            print("Rotation: ", self.rotationMatrixToEulerAngles(R))

        for T in Ts:
            print("Translation: ", T)

        for N in Ns:
            print("Normls: ", N)

        height_panorama = img1.shape[0]+500
        width_panorama = img1.shape[1]+500

        panorama1 = np.zeros((height_panorama, width_panorama, 3))

        panorama1[0: (img1.shape[0]), 0: img1.shape[1], :] = img1

        panorama2 = cv2.warpPerspective(
            img2, H, (width_panorama, height_panorama))

        result = panorama1+panorama2
        cv2.normalize(result,  result, 0, 255, cv2.NORM_MINMAX)
        return result


if __name__ == '__main__':
    img2 = cv2.imread('right_concrete.jpg')
    global calculated_angle
    img1 = cv2.imread('left_concrete.jpg')
    img3 = cv2.imread('left_concrete_left.jpg')
    final = Image_Stitching().blending(img1, img2, img3)
    cv2.imwrite('stitch.jpg', final)

    print(img2.shape[1])

    plt.plot([cam1_x, cam1_x-l*math.tan(camera_angle/2)],
             [0, l], color='black')
    plt.plot([cam1_x, cam1_x+l*math.tan(camera_angle/2)],
             [0, l], color='black')

    plt.plot([cam2_x, cam2_x -
              l*math.tan(camera_angle/2+cam2_angle)], [0, l], color='black')
    plt.plot([cam2_x, cam2_x +
              l*math.tan(camera_angle/2-cam2_angle)], [0, l], color='black')

    plt.xlim(-1, 1)
    plt.ylim(0, 2)

    plt.grid(linestyle='dotted')

    plt.savefig('cross.eps', format='eps')
