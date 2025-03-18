class Review:
  def __init__(self, id, review_body, rating_value):
    self.id = id
    self.review_body = review_body
    self.review_value = rating_value

  def __repr__(self):
    return f'<Review {self.id}>'
  
  def serialize(self):
    return {
      'id': self.id
    }