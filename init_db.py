from db import Base, engine
import models

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created.")
