import streamlit as st
from evaluate_func import *
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
    select = st.radio('choice', path_list.keys())
    image = st.selectbox('path_list', path_list[select])
    cv_img, pil_img = load_image(image)

    fd = inference(image)
    print(fd)
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, width=500)

    # faces_bbox = request_func(cv_img)

    with col2:
        for _, k in fd.items():
            # for _, v in k.items():
            if len(k) == 0:
                continue
            for face in k:
                bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
                cv2.rectangle(cv_img, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 0, 255), 3)
            st.image(Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)), width=500)
            st.write(f'검출된 얼굴 {len(k)}개')
    if st.button('click'):
        st.text(calculate_map_score({select:fd}, './src/data/ground_truth/wider_medium_val.json'))


if __name__ == '__main__':
    test()