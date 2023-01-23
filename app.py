import streamlit as st
import streamlit.components.v1 as stc
import lgbm


def main():

    st.title("Qiitaいいね数予測アプリ")

    title = st.text_input(label="記事のタイトルを入力して下さい")
    body = st.text_area(label="記事の本文を入力して下さい", height=500)
    tags = st.text_input(label="タグをカンマ区切りで入力して下さい")
    followers_count = st.number_input(label="フォロワー数を入力して下さい", step=1, min_value=0)
    organization = st.radio(label="組織に参加していますか", options=["はい", "いいえ"])
    items_count = st.number_input(label="既に公開済みの記事の数を入力して下さい", step=1, min_value=0)

    pred_button = st.button("いいね数を予測する")

    analyzer = lgbm.Analyzer()

    if pred_button is True:
        pred = analyzer.predict(title, body, tags, followers_count, organization, items_count)
        st.write(f"## いいね数の予測値は {pred:.2f} です！")


if __name__ == "__main__":
    main()