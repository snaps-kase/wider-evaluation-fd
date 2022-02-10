import streamlit as st
from demo import *
import cv2
from PIL import Image


def load_image(img_path):
    cv_img = cv2.imread(img_path)
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return cv_img, pil_img

def test():
    st.set_page_config(layout="wide")
    st.sidebar.title('사용하실 API를 선택하세요.')

    path_list = load_data('./wider_val')
    image = st.selectbox('path_list', path_list)
    cv_img, pil_img = load_image(image)

    fd = request_fd_api(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, width=500)

    # faces_bbox = request_func(cv_img)

    with col2:
        for _, k in fd.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                for face in v:
                    bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
                    cv2.rectangle(cv_img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 0, 255), 3)
                st.image(Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)), width=500)
                st.write(f'검출된 얼굴 {len(v)}개')
    if st.button('click'):
        st.write(calculate_score(fd,'./ground_truth'))



if __name__ == '__main__':
    test()