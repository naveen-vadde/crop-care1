"""
Check the MongoDB Atlas connection and create the indexes.

    python init_db.py            # test connection + create indexes (safe, keeps data)
    python init_db.py --reset    # DELETE all CropCare data, then recreate indexes (asks first)
"""
import sys

import config
import database


def main():
    try:
        db = database.init_db()
    except RuntimeError as exc:
        print(f"\n❌ {exc}\n")
        sys.exit(1)

    if '--reset' in sys.argv:
        answer = input(f"This will DELETE ALL DATA in database '{config.MONGODB_DB}'. Type 'yes' to continue: ")
        if answer.strip().lower() != 'yes':
            print("Aborted.")
            sys.exit(1)
        for name in ('disease_records', 'plants', 'users'):
            db[name].drop()
            print(f"Dropped collection: {name}")
        database.create_indexes()

    print(f"✅ Connected to MongoDB Atlas - database '{config.MONGODB_DB}'")
    names = db.list_collection_names()
    print("Collections:", ", ".join(sorted(names)) if names else "(none yet - created on first use)")
    print("Indexes are ready.")


if __name__ == '__main__':
    main()
