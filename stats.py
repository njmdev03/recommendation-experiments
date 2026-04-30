import matplotlib.pyplot as plt
import seaborn as sns

import config
import utils


raw_news, raw_behaviors = utils.get_dataset()

print("Example rows")
print(f"News:")
print(raw_news["N55528"])
# print(raw_news.news_df)
print()
print(f"Beh")
print(raw_behaviors[0])
# print(raw_behaviors.behaviors_df)
print()

# Basic statistics
print("Aggregate Stats")
print(f'Total users: {raw_behaviors.behaviors_df["user_id"].nunique()}')
print(f'Total news articles: {len(raw_news.news_df)}')
print(f'Total impressions: {len(raw_behaviors.behaviors_df)}')

print(f'Total Categories: {raw_news.news_df['category'].nunique()}')

# Category distribution
cat_counts = raw_news.news_df['category'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=cat_counts.index, y=cat_counts.values)
plt.title('Article Count by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('out/category_distribution.png')

# # User click distribution
# def count_clicks(imp_str):
#     if pd.isna(imp_str): return 0
#     return sum(1 for x in imp_str.split() if x.endswith('-1'))

# behaviors_df['n_clicks'] = behaviors_df['impressions'].apply(count_clicks)
# user_clicks = behaviors_df.groupby('user_id')['n_clicks'].sum()
# plt.figure(figsize=(10, 5))
# plt.hist(user_clicks, bins=50, edgecolor='black')
# plt.xlabel('Total Clicks per User')
# plt.ylabel('Number of Users')
# plt.title('Distribution of User Click Activity')
# plt.savefig('results/user_click_dist.png')
