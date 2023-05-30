import streamlit as st
import altair as alt

import download_corpora
from utils import SentimentAnalysis, SpellingCorrection, PartsOfSpeechTagging, TextSummarize
from sample_text import text_input, what_is_nlp




st.title('📝 Natural language processing')
st.caption('''Natural language processing (NLP) refers to the branch of computer science and more specifically, the branch of artificial intelligence or AI concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.''')

tab1, tab2, tab3, tab4 = st.tabs(
    tabs=[':blue[Sentiment Analysis]', ':green[Spelling Correction]', ':orange[Part-of-speech Tagging]', ':violet[Text Summarization]'])



with tab1:
    st.header('Sentiment Analysis')
    text_to_analyze = st.text_area(label="Enter the text to analyse", value=text_input)
    if text_to_analyze:
        sentiment = SentimentAnalysis(text=text_to_analyze)
        response = sentiment.get_mood(threshold=0.3)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Sentiment Score", value=response.emoji, delta=round(response.sentiment,2))
            st.divider()
            c = alt.Chart(sentiment.get_word_count_df()).mark_circle().encode(x='words', y='count', size='count', color='count')
            st.altair_chart(altair_chart=c, use_container_width=True)
            
        with col2:
            analysis = sentiment.polarity_and_subjectivity()
            st.caption(analysis)

with tab2:
    st.header('Spelling Correction')
    text_to_spell_check = st.text_input(label="Enter the text to correct", value='I acknowlege that it is about 70% acurate')
    if text_to_spell_check:
        spellCheck = SpellingCorrection(text=text_to_spell_check)
        response = spellCheck.spelling_correction()
        st.write(response)
        st.caption(spellCheck.redlines(), unsafe_allow_html=True)

        st.divider()

        explanation, keys = spellCheck.get_explanation()
        
        selected_keys = st.multiselect(
            label=  "Spelling correction found" if keys else 'No correction required', 
            options=keys, max_selections=5, 
            default=keys[:4] if keys else None)
        
        for key in selected_keys:
            response = explanation[key]

            with st.expander(label=f':violet[Correction for {key}]', expanded=True):
                for suggestion, confidence in response['correction']:
                    st.caption(f'📢 :orange[Suggestion] : :green[{suggestion}] 🔢 :blue[confidence] : {round(confidence,2)}')

                if response['definition']:
                    st.subheader(f':violet[Definition] of {response["correction"][0][0]}')
                    for defs in response['definition']:
                        st.code(defs)





with tab3:
    st.header('Part-of-speech Tagging')
    text_to_tag = st.text_input("Enter the text you want to tag", value='I work at Google and this is my sentence.')
    if text_to_tag:
        tagging = PartsOfSpeechTagging(text=text_to_tag)
        df = tagging.parts_of_speech_tagging()
        st.write(df)
        
        tag_map = tagging.parts_of_speech_map()
        if tag_map:
            keys = st.multiselect(label="Part-of-speech", options=list(tag_map.keys()), default=list(tag_map.keys())[:3], max_selections=5)
                
            for key in keys:
                try:
                    name = tagging.part_of_speech_help()[key]
                except:
                    name = ''
                with st.expander(label=f":orange[{key}] | :green[{name}]", expanded=True):
                    for word in tag_map[key]:
                        st.code(word)


with tab4:
    st.header("Text Summarization")
    text_to_summarize = st.text_area(
        label="Enter the text you want to summarize", value=what_is_nlp.strip())
    if text_to_summarize:
        summarizer = TextSummarize(text=text_to_summarize)
        num = st.number_input(
            label="How many sentences do you want in the summary?", min_value=1, step=1, value=3)
        markup, summary = summarizer.get_summary(n=num)
        st.subheader('Summary')

        st.caption(summary)
        
        with st.expander(label=":orange[Highlights]", expanded=False):
            st.caption(markup)

        tokens = summarizer.get_token_frequency_df()
        token_chart = alt.Chart(data=tokens).mark_circle().encode(x='Tokens', y='Weights', size='Weights', color='Weights')
        st.altair_chart(altair_chart=token_chart, use_container_width=True)



