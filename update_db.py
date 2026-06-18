import sqlite3

def update_database():
    try:
        # Connect to your existing database file
        conn = sqlite3.connect('hoams.db')
        cursor = conn.cursor()

        print("Connecting to hoams.db...")

        # Add token_number column to appointments table
        try:
            cursor.execute("ALTER TABLE appointments ADD COLUMN token_number INTEGER")
            print("✅ Column 'token_number' added.")
        except sqlite3.OperationalError:
            print("ℹ️ 'token_number' already exists.")

        # Add status column to appointments table
        try:
            cursor.execute("ALTER TABLE appointments ADD COLUMN status TEXT DEFAULT 'Scheduled'")
            print("✅ Column 'status' added.")
        except sqlite3.OperationalError:
            print("ℹ️ 'status' already exists.")

        conn.commit()
        conn.close()
        print("\n🎉 Database update complete!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    update_database()