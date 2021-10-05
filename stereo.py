import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

camera_angle = np.radians(40)
cam1_x = -0.2
cam2_x = 0.2
sensor_size = 0.035
cam2_angle = np.radians(0.0)

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
        self.ratio = 0.9
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

        # cv2.normalize(H, H)

        x_ = []
        y_ = []

        num_cols = img1.shape[1]
        num_rows = img2.shape[0]

        for point1, point2 in zip(image1_kp, image2_kp):

            angle1 = camera_angle*((point1[0])/img1.shape[1]-0.5)
            angle2 = camera_angle*((point2[0])/img2.shape[1]-0.5)

            line1 = [[cam1_x, 0],
                     [cam1_x + (l)*math.tan(angle1), l]]
            line2 = [[cam2_x, 0], [
                cam2_x + (l)*math.tan(angle2-cam2_angle), l]]

            intersect = line_intersection(line1, line2)

            plt.scatter(intersect[0], intersect[1], 1.5, linewidths="0.0")
            # plt.plot(line1, color='red')
            # plt.plot(line2, color='green')

        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
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
        self.registration(img1, img2, img3)

        return img1


if __name__ == '__main__':
    img2 = cv2.imread('dots.png')
    global calculated_angle
    img1 = cv2.imread('dots_left.png')
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

    plt.xlim(-3, 3)
    plt.ylim(0, 7)

    plt.grid(linestyle='dotted')

    plt.savefig('cross.eps', format='eps')
