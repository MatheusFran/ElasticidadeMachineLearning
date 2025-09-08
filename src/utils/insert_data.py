import pandas as pd
import psycopg2

def insert_data():
    df = pd.read_csv('../../data/raw/zara.csv', encoding='utf-8',sep=';')

    # Padroniza nomes das colunas
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Converte booleans
    df["promotion"] = df["promotion"].map({"Yes": True, "No": False, "1": True, "0": False})
    df["seasonal"] = df["seasonal"].map({"Yes": True, "No": False, "1": True, "0": False})
    df = df.drop(columns=["sku"],axis=1)
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
            CREATE TABLE IF NOT EXISTS zara (
                product_id TEXT PRIMARY KEY,
                product_position TEXT,
                promotion BOOLEAN,
                product_category TEXT,
                seasonal BOOLEAN,
                sales_volume INT,
                brand TEXT,
                url TEXT,
                name TEXT,
                description TEXT,
                price NUMERIC(10,2),
                currency TEXT,
                scraped_at TIMESTAMP,
                terms TEXT,
                section TEXT
            );
        """)
        conn.commit()

        rows = df.to_records(index=False).tolist()
        cols = ','.join(df.columns)
        placeholders = ','.join(['%s'] * len(df.columns))

        insert_query = f"INSERT INTO zara ({cols}) VALUES ({placeholders}) ON CONFLICT (product_id) DO NOTHING"
        cursor.executemany(insert_query, rows)
        conn.commit()

        print(f"✅ Inseridos {cursor.rowcount} registros na tabela 'zara'.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"❌ Erro: {error}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == '__main__':
    insert_data()
