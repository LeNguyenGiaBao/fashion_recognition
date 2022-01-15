import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from fashion_detection.model import SSD, Predictor
from utils.utils import draw_boxes, cut_cothes
from utils.color_recognition import get_color
st.set_page_config(page_title='Fashion Recognition',page_icon="👗", layout="wide")

@st.cache
def load_model(model_path, class_names):
    net = SSD(len(class_names), is_test=True)
    net.load(model_path)
    model = Predictor(net, candidate_size=200)
    
    return model

def predict_od(model, img):
    boxes, labels, probs = model.predict(img, 50, thresh)
    list_clothes = cut_cothes(img, boxes, labels, probs, class_names)
    img_draw = draw_boxes(img, boxes, labels, probs, class_names)

    return img_draw, list_clothes

def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def UI():
    st.markdown("<h1 style='text-align: center; color: black;'>Fashion Recognition</h1>", unsafe_allow_html=True)
    
    _, a, _ = st.columns((1,10, 1))
    a.markdown("<p style='text-align: left; color: black; font-size: 25px;'>A website where you can find all fashion styles in world with only a picture</p>", unsafe_allow_html=True)
    
    # _, a, _ = st.columns((1,10,1))
    img_file_buffer = a.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
    if img_file_buffer is not None:
        img = np.array(Image.open(img_file_buffer))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_draw, list_clothes = predict_od(model_od, img)

        # _, a, _ = st.columns((1,10, 1))
        a.image(img_draw)
        a.markdown('---')

        _, a, b, _ = st.columns((1,3,7, 1))
        for i in list_clothes:
            _, a, b, _ = st.columns((1,2,3, 1))
            img_clothes = i['img']
            label_clothes = i['label']
            color_clothes = get_color(img_clothes, 4)
            a.image(img_clothes)
            b.markdown('## {}'.format(label_clothes))
            b.markdown('### Color: {}'.format(color_clothes))
            _, a, _ = st.columns((1,5, 1))
            a.markdown('---')




    # st.markdown('---')
    # _, a = st.columns((1,10))
    # a.markdown("## About Us")
    # st.markdown('')
    # _, a1, _, a2, _, a3, _ = st.columns((1,3,1,3,1,3,1))
    # a1.markdown('<p align="center"><img src="https://avatars.githubusercontent.com/u/68860804?v=4" width="300px height="300px" /></p>', unsafe_allow_html=True)
    # a1.markdown("<h2 style='text-align: center; color: black;'>Lê Nguyễn Gia Bảo</h1>", unsafe_allow_html=True)
    # a1.markdown('''
    # <p align="center">
    #     <a href="https://www.linkedin.com/in/lenguyengiabao/" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/>
    #     </a>
    #     <a href="https://www.facebook.com/baorua.98/" alt="Facebook" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" />
    #     </a> 
    #     <a href="https://github.com/LeNguyenGiaBao" alt="Github" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/github.png"/>
    #     </a> 
    #     <a href="https://www.youtube.com/channel/UCOZbUfO_au3oxHEh4x52wvw/videos" alt="Youtube channel" target="_blank" >
    #         <img src="https://img.icons8.com/fluent/48/000000/youtube-play.png"/>
    #     </a>
    #     <a href="https://www.kaggle.com/nguyngiabol" alt="Kaggle" target="_blank" >
    #         <img src="https://img.icons8.com/windows/48/000000/kaggle.png"/>
    #     </a>
    #     <a href="mailto:lenguyengiabao46@gmail.com" alt="Email" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/mailing.png"/>
    #     </a>
    # </p>
    # ''', unsafe_allow_html=True)

    # a2.markdown('<p align="center"><img src="https://scontent.fsgn4-1.fna.fbcdn.net/v/t1.6435-9/52350962_2309926179240286_530087548026880000_n.jpg?_nc_cat=103&ccb=1-5&_nc_sid=174925&_nc_ohc=M8o02dbKO1oAX_9B3GG&_nc_ht=scontent.fsgn4-1.fna&oh=00_AT-emuEA16gPfdz2rGK7vyX-1jPGnKR1pfVbohZE-GiJzA&oe=6208102C" width="300px height="300px" /></p>', unsafe_allow_html=True)
    # a2.markdown("<h2 style='text-align: center; color: black;'>Lê Bảo Khanh</h1>", unsafe_allow_html=True)
    # a2.markdown('''
    # <p align="center">
    #     <a href="https://www.linkedin.com/in/lenguyengiabao/" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/>
    #     </a>
    #     <a href="https://www.facebook.com/trantrungkien2035" alt="Facebook" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" />
    #     </a> 
    #     <a href="https://github.com/ttkien2035" alt="Github" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/github.png"/>
    #     </a> 
    #     <a href="https://www.youtube.com/channel/UCOZbUfO_au3oxHEh4x52wvw/videos" alt="Youtube channel" target="_blank" >
    #         <img src="https://img.icons8.com/fluent/48/000000/youtube-play.png"/>
    #     </a>
    #     <a href="https://www.kaggle.com/nguyngiabol" alt="Kaggle" target="_blank" >
    #         <img src="https://img.icons8.com/windows/48/000000/kaggle.png"/>
    #     </a>
    #     <a href="mailto:trantrungkien2035@gmail.com" alt="Email" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/mailing.png"/>
    #     </a>
    # </p>
    # ''', unsafe_allow_html=True)

    # a3.markdown('<p align="center"><img src="https://scontent.fsgn4-1.fna.fbcdn.net/v/t1.6435-9/187385484_1388017121598199_993741695716895236_n.jpg?_nc_cat=101&ccb=1-5&_nc_sid=8bfeb9&_nc_ohc=w_V_PMkcD-cAX_-0YhN&_nc_ht=scontent.fsgn4-1.fna&oh=00_AT86Mg5AlWdToW51nH_2iGL50lbtDgxmhLcLnYWsWqC0WA&oe=6205531F" width="300px height="300px" /></p>', unsafe_allow_html=True)
    # a3.markdown("<h2 style='text-align: center; color: black;'>Nguyễn Thị Yến Nhi", unsafe_allow_html=True)
    # a3.markdown('''
    # <p align="center">
    #     <a href="https://www.linkedin.com/in/lenguyengiabao/" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/>
    #     </a>
    #     <a href="https://www.facebook.com/trantrungkien2035" alt="Facebook" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/facebook-new.png" />
    #     </a> 
    #     <a href="https://github.com/ttkien2035" alt="Github" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/github.png"/>
    #     </a> 
    #     <a href="https://www.youtube.com/channel/UCOZbUfO_au3oxHEh4x52wvw/videos" alt="Youtube channel" target="_blank" >
    #         <img src="https://img.icons8.com/fluent/48/000000/youtube-play.png"/>
    #     </a>
    #     <a href="https://www.kaggle.com/nguyngiabol" alt="Kaggle" target="_blank" >
    #         <img src="https://img.icons8.com/windows/48/000000/kaggle.png"/>
    #     </a>
    #     <a href="mailto:trantrungkien2035@gmail.com" alt="Email" target="_blank">
    #         <img src="https://img.icons8.com/fluent/48/000000/mailing.png"/>
    #     </a>
    # </p>
    # ''', unsafe_allow_html=True)

if __name__ == "__main__":
    thresh = 0.5
    class_names = ['BACKGROUND', 'sunglass', 'hat', 'jacket', 'shirt', 'pants', 'shorts', 'skirt', 'dress', 'bag', 'shoe']
    model_od = load_model('./models/vgg16-ssd-Epoch-125-Loss-2.8042236075681797.pth', class_names)
    
    UI()
