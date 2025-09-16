import pandas as pd
import psycopg2
import uuid


def insert_data():
    df = pd.read_csv('../../data/raw/scanner_data.csv', encoding='utf-8', index_col=0)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.strftime('%Y-%m-%d')

    try:
        conn = psycopg2.connect(
            database="elasticidade",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS store (
                id TEXT PRIMARY KEY,
                date DATE,
                customer_id TEXT,
                transaction_id TEXT,
                sku_category TEXT,
                sku TEXT,
                quantity NUMERIC,
                sales_amount NUMERIC
            )
        """)
        conn.commit()

        rows = df.to_records(index=False).tolist()
        cols = ','.join(df.columns)
        placeholders = ','.join(['%s'] * len(df.columns))

        insert_query = f"""
            INSERT INTO store ({cols}) 
            VALUES ({placeholders}) 
            ON CONFLICT (id) DO NOTHING
        """
        cursor.executemany(insert_query, rows)
        conn.commit()

        print(f"✅ Inseridos {cursor.rowcount} registros na tabela 'store'.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"❌ Erro: {error}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == '__main__':
    insert_data()
