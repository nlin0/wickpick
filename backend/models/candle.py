'''
"1": {
        "name": "Lucky Shamrock™ - Returning Favorite",
        "category": "Fresh & Clean",
        "company": "Yankee Candlee",                    
        "link": "https://www.yankeecandle.com/yankee-candle/candles/candle-styles/original-jar-candles/lucky-shamrock---returning-favorite/ORCL_1632868.html",
        "description": "This is a Fresh & Clean fragrance. A touch of the Emerald Isle…the fresh scent of lush, green hills kissed by a sparkle of sunshine. Top notes: Green Notes, Zesty Citrus. Middle notes: Eucalyptus, Hyacinth. Base notes: Cedar Wood, Violet Leaf. Top note is the initial impression of the fragrance, middle note is the main body of the scent and base is its final impression.",
        "overall_rating": "4.3",
        "overall_reviewcount": "674",
        "img_url": "images/1.jpg",
        "reviews": {
            "review1": {
                "review_body": "This is one of my favorite scents! It is so clean and fresh and has a great throw",
                "rating_value": 5
            },
            "review2": {
                "review_body": "Lovely grassy scent, nice and clean. Lightly wet, fresh-cut grass. It's a classic for a reason.",
                "rating_value": 5
            },
            "review3": {
                "review_body": "I discovered this candle last year and enjoyed it so much that I bought it again to burn in March. Its fresh green scent is perfect for the end of winter, when your spirits need a lift.",
                "rating_value": 5
            },
            "review4": {
                "review_body": "This is my third Lucky Shamrock candle. I really enjoy the natural sence. It's very subtle and pleasant.",
                "rating_value": 5
            },
            "review5": {
                "review_body": "I waited all year for this scent to come out. I am originally from Ireland and this reminds me so much of home. I usually buy as many as they have in stock. If you want, the original smells of Ireland, this candle is for you.",
                "rating_value": 5
            },
            "review6": {
                "review_body": "I love seeing the classic seasonal fragrances of the past return for limited engagements.  What I would really like to see are the returning favorites be offered in the newer soy blend wax but still in the original apothecary jars.  While paraffin was the original formula and will always be a classic, it burns too slowly, which doesn't allow us to rotate out our fragrances as often as we'd like.",
                "rating_value": 4
            },
            "review7": {
                "review_body": "I loved the jar to match my decorations. Also enjoyed the scent. Unfortunately it didnt burn evenly. I endednup with a hole in the middle and hard wax on the sides.",
                "rating_value": 2
            },
            "review8": {
                "review_body": "Fresh scent...will repurchase. Each scent so far is pretty well saturated instead of just on the surface like other candles. Scent remains continuous throughout burn time. Love that about Yankee.",
                "rating_value": 4
            }
        }
    }
'''
class Candle:
  def __init__(self, id, name, category, link, description, overall_rating, overall_reviewcount, img_url, reviews):
    self.id = id
    self.name = name
    self.category = category
    self.link = link
    self.description = description
    self.overall_rating = overall_rating
    self.overall_reviewcount = overall_reviewcount
    self.img_url = img_url
    self.reviews = reviews

  def __repr__(self):
    return f'<Candle {self.id}>'
  
  def serialize(self):
    return {
      'id': self.id
    }