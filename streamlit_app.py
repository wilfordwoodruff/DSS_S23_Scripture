import streamlit as st
import pandas as pd
import altair as alt

# load dataset
data = pd.read_csv("/Users/larenawaddell/Desktop/DSS/DSS_S23_Scripture/data/data_matches.csv") 

# title and intro of app
st.write("## Wilford Woodruff's Scripture References")
st.write("The scripture references in Wilford Woodruff's journal entries were found by using the TF-IDF vectorizer, which assigns weights to words in a document based on their frequency and rarity across journal entries and scriptures. TF-IDF captures the importance of a word by considering both how often it appears in a document (TF) and how rare it is in the entire corpus (IDF), allowing for meaningful representation of text data. ")
st.write("To quantify how well the scripture references match, we use cosine similarity. Cosine similarity calculates the cosine of the angle between vectors, which represents the similarity between their directions. A value of 1 indicates that the vectors are identical, 0 indicates no similarity, and -1 indicates they are completely dissimilar.")
st.write("For the purposes of this app, we only show matches with a cosine similarity of at least 0.7. Use the select box below to filter to a gospel in the standard works.")


# volume title options
volume_title = ['All'] + sorted(data['volume_title'].unique())

# select volume title to filter
selected_volume_title = st.selectbox("Select a Gospel", volume_title)

# filter data based on selection
if selected_volume_title != 'All':
    filtered_data = data[(data['volume_title'] == selected_volume_title)]
else:
    filtered_data = data

# display filtered data
st.write("### First 10 Matches Over .70 Cosine Similarity")
st.dataframe(filtered_data.filter(['cosine_score', 'text_woodruff', 'text_scriptures', 'verse_title', 'volume_title', 'dates', 'internal_id', 'parent_id', 'order', 'website_url']).head(10))

# summary
st.write("Number of matches:", len(filtered_data))
st.write("Average cosine similarity:", filtered_data['cosine_score'].mean())

# separate multiple dates and convert them to datetime
filtered_data['dates'] = filtered_data['dates'].str.split("\|")
filtered_data['dates'] = filtered_data['dates'].apply(pd.to_datetime)

# create df for graph
plot_data = pd.DataFrame({'dates': filtered_data['dates'].explode()})

# group by date and count the occurrences
plot_data['count'] = plot_data.groupby('dates').cumcount() + 1

# altair date chart
date_chart = (alt.Chart(plot_data)
              .mark_line(point=True)
              .encode(
                  x=alt.X('dates:T', title='Date'),
                  y=alt.Y('count:Q', title='Number of Matches'),
                  tooltip=['dates:T', 'count:Q']
    )
)

# display the date chart
st.write("### Woodruff's Scripture References Over Time")
st.altair_chart(date_chart, use_container_width=True)

# altair boxplot of cosine similarities in each gospel
box_chart = (alt.Chart(data)
             .mark_boxplot()
             .encode(
                 x=alt.X('volume_title', title='Gospel'), 
                 y=alt.Y('cosine_score', title='Cosine Similarity', scale=alt.Scale(domain=[.7, 1])), 
                 tooltip=['dates:T', 'verse_title']
                 )
             )

# display the boxplot
st.write("### Distribution of Cosine Score in All Gospels")
st.altair_chart(box_chart, use_container_width=True)


