import sqlite3

from fantasy_baseball_manager.domain.statcast_pitch import StatcastPitch


class SqliteStatcastPitchRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, pitch: StatcastPitch) -> int:
        cursor = self._conn.execute(
            """INSERT INTO statcast_pitch
                   (game_pk, game_date, batter_id, pitcher_id,
                    at_bat_number, pitch_number, pitch_type,
                    release_speed, release_spin_rate,
                    pfx_x, pfx_z, plate_x, plate_z, zone,
                    events, description,
                    launch_speed, launch_angle, hit_distance_sc,
                    barrel, estimated_ba_using_speedangle,
                    estimated_woba_using_speedangle, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_pk, at_bat_number, pitch_number) DO UPDATE SET
                   game_date=excluded.game_date,
                   batter_id=excluded.batter_id,
                   pitcher_id=excluded.pitcher_id,
                   pitch_type=excluded.pitch_type,
                   release_speed=excluded.release_speed,
                   release_spin_rate=excluded.release_spin_rate,
                   pfx_x=excluded.pfx_x, pfx_z=excluded.pfx_z,
                   plate_x=excluded.plate_x, plate_z=excluded.plate_z,
                   zone=excluded.zone,
                   events=excluded.events, description=excluded.description,
                   launch_speed=excluded.launch_speed,
                   launch_angle=excluded.launch_angle,
                   hit_distance_sc=excluded.hit_distance_sc,
                   barrel=excluded.barrel,
                   estimated_ba_using_speedangle=excluded.estimated_ba_using_speedangle,
                   estimated_woba_using_speedangle=excluded.estimated_woba_using_speedangle,
                   loaded_at=excluded.loaded_at""",
            (
                pitch.game_pk,
                pitch.game_date,
                pitch.batter_id,
                pitch.pitcher_id,
                pitch.at_bat_number,
                pitch.pitch_number,
                pitch.pitch_type,
                pitch.release_speed,
                pitch.release_spin_rate,
                pitch.pfx_x,
                pitch.pfx_z,
                pitch.plate_x,
                pitch.plate_z,
                pitch.zone,
                pitch.events,
                pitch.description,
                pitch.launch_speed,
                pitch.launch_angle,
                pitch.hit_distance_sc,
                pitch.barrel,
                pitch.estimated_ba_using_speedangle,
                pitch.estimated_woba_using_speedangle,
                pitch.loaded_at,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_pitcher_date(self, pitcher_id: int, game_date: str) -> list[StatcastPitch]:
        rows = self._conn.execute(
            "SELECT * FROM statcast_pitch WHERE pitcher_id = ? AND game_date = ?",
            (pitcher_id, game_date),
        ).fetchall()
        return [self._row_to_pitch(row) for row in rows]

    def get_by_batter_date(self, batter_id: int, game_date: str) -> list[StatcastPitch]:
        rows = self._conn.execute(
            "SELECT * FROM statcast_pitch WHERE batter_id = ? AND game_date = ?",
            (batter_id, game_date),
        ).fetchall()
        return [self._row_to_pitch(row) for row in rows]

    def get_by_game(self, game_pk: int) -> list[StatcastPitch]:
        rows = self._conn.execute(
            "SELECT * FROM statcast_pitch WHERE game_pk = ?",
            (game_pk,),
        ).fetchall()
        return [self._row_to_pitch(row) for row in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM statcast_pitch").fetchone()
        return row[0]

    @staticmethod
    def _row_to_pitch(row: tuple) -> StatcastPitch:
        return StatcastPitch(
            id=row[0],
            game_pk=row[1],
            game_date=row[2],
            batter_id=row[3],
            pitcher_id=row[4],
            at_bat_number=row[5],
            pitch_number=row[6],
            pitch_type=row[7],
            release_speed=row[8],
            release_spin_rate=row[9],
            pfx_x=row[10],
            pfx_z=row[11],
            plate_x=row[12],
            plate_z=row[13],
            zone=row[14],
            events=row[15],
            description=row[16],
            launch_speed=row[17],
            launch_angle=row[18],
            hit_distance_sc=row[19],
            barrel=row[20],
            estimated_ba_using_speedangle=row[21],
            estimated_woba_using_speedangle=row[22],
            loaded_at=row[23],
        )
