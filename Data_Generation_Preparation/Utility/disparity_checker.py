import cv2
import numpy as np

class StereoDepthVisualizer:
    def __init__(self, left_img_path, right_img_path, focal_length, baseline):
        self.left_img = cv2.imread(left_img_path)
        self.right_img = cv2.imread(right_img_path)
        self.focal_length = focal_length
        self.baseline = baseline
        self.left_points = []
        self.right_points = []
        self.current_image = 'left'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_image == 'left':
                self.left_points.append((x, y))
                cv2.circle(self.left_img, (x, y), 5, (0, 255, 0), -1)
                if len(self.left_points) == 2:
                    self.current_image = 'right'
                    print("Now click on the corresponding points in the right image.")
            elif self.current_image == 'right':
                self.right_points.append((x, y))
                cv2.circle(self.right_img, (x, y), 5, (0, 255, 0), -1)
                if len(self.right_points) == 2:
                    self.calculate_and_draw()

    def calculate_and_draw(self):
        # Calculate disparities
        disparities = [abs(lp[0] - rp[0]) for lp, rp in zip(self.left_points, self.right_points)]
        
        # Calculate depths
        depths = [(self.focal_length * self.baseline) / d for d in disparities]
        
        # Calculate 3D coordinates
        points_3d = []
        for i in range(2):
            z = depths[i]
            x = (self.left_points[i][0] - self.left_img.shape[1]/2) * z / self.focal_length
            y = -(self.left_points[i][1] - self.left_img.shape[0]/2) * z / self.focal_length
            points_3d.append((x, y, z))
        
        # Calculate distance
        distance = np.sqrt(sum((a-b)**2 for a, b in zip(points_3d[0], points_3d[1])))
        
        # Draw line and display distance on left image
        cv2.line(self.left_img, self.left_points[0], self.left_points[1], (0, 0, 255), 2)
        mid_point = ((self.left_points[0][0] + self.left_points[1][0]) // 2, 
                     (self.left_points[0][1] + self.left_points[1][1]) // 2)
        cv2.putText(self.left_img, f"{distance:.2f}", mid_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        print(f"Distance between points: {distance:.2f} units")
        print("You can select two new points in the left image.")
        self.current_image = 'left'
        self.left_points = []
        self.right_points = []

    def run(self):
        cv2.namedWindow('Left Image')
        cv2.namedWindow('Right Image')
        cv2.setMouseCallback('Left Image', self.mouse_callback)
        cv2.setMouseCallback('Right Image', self.mouse_callback)

        print("Click on two points in the left image.")

        while True:
            cv2.imshow('Left Image', self.left_img)
            cv2.imshow('Right Image', self.right_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    idx = "1252"
    left_img_path = "/home/utsav/IProject/data/captured/lnd1/rect_left/{}.png".format(idx)
    right_img_path = "/home/utsav/IProject/data/captured/lnd1/rect_right/{}.png".format(idx)
    focal_length = 847.28300117  # camera's focal length in pixels
    baseline = 5.5160  # stereo camera baseline in meters

    verifier = StereoDepthVisualizer(left_img_path, right_img_path, focal_length, baseline)
    verifier.run()