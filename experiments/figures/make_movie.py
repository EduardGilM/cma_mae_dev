import os
import cv2

image_folder = '../lin_proj/logs/cma_mae_100_1.0/trial_0/'
video_name = 'video.avi'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

# # adapted from https://stackoverflow.com/questions/57104921/cv2-addweighted-except-some-color
# # [17,14,8] is background color for slime volleyball
# def overlay_two_image(image, overlay, ignore_color=[17,14,8]):
#     ignore_color = np.asarray(ignore_color)
#     mask = ~(overlay==ignore_color).all(-1)
#     # Or mask = (overlay!=ignore_color).any(-1)
#     out = image.copy()
#     out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
#     return out

# def make_overlay(movie_path, save_overlay_to, num_overlay, interactive=False):
#     movie = cv2.VideoCapture(movie_path)
#     num_frames = movie.get(cv2.CAP_PROP_FRAME_COUNT)

#     # final frame is the background, so read from end to start
#     overlay_frames = np.arange(num_frames, step=20)[::-1] if interactive else np.linspace(num_frames-1, 0, num_overlay, dtype='int')
#     for frame_id in overlay_frames:
#         movie.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#         ret, frame = movie.read()

#         if ret:
#             if 'overlay' in vars():
#                 if interactive:
#                     cv2.imshow('', frame)
#                     if cv2.waitKey() == ord('y'):
#                         overlay = overlay_two_image(overlay, frame)
#                 else:
#                     overlay = overlay_two_image(overlay, frame)
#             else:
#                 overlay = frame

#     cv2.imwrite(save_overlay_to, overlay)

# make_overlay('video.mp4', 'overlay.png', 8, False)