from __future__ import annotations

import argparse
from datetime import UTC, datetime

import web_api


def create_or_update_admin(username: str, password: str, email: str | None = None) -> None:
    username = username.strip()
    if not username:
        raise ValueError("username is required")
    if len(password.strip()) < 6:
        raise ValueError("password must be at least 6 characters")

    db = web_api.SessionLocal()
    try:
        row = db.query(web_api.User).filter(web_api.User.username == username).first()
        if row is None:
            row = web_api.User(
                username=username,
                email=email.strip() if email else None,
                password_hash=web_api.hash_password(password),
                is_admin=True,
                role="admin",
                created_at=datetime.now(UTC),
                last_login_at=None,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            print(f"[ok] created admin user: {row.username} (id={row.id})")
            return

        row.password_hash = web_api.hash_password(password)
        row.is_admin = True
        row.role = "admin"
        if email is not None:
            row.email = email.strip() or None
        db.commit()
        print(f"[ok] updated admin user: {row.username} (id={row.id})")
    finally:
        db.close()


def list_admins() -> None:
    db = web_api.SessionLocal()
    try:
        rows = db.query(web_api.User).filter(web_api.User.is_admin == True).all()  # noqa: E712
        if not rows:
            print("[info] no admin users found.")
            return
        for row in rows:
            print(
                f"id={row.id} username={row.username} email={row.email or '-'} "
                f"role={row.role or 'user'} created_at={row.created_at.isoformat() if row.created_at else '-'}"
            )
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Admin account management for Flood Web API.")
    sub = parser.add_subparsers(dest="command", required=True)

    create_cmd = sub.add_parser("create", help="Create or update an admin user.")
    create_cmd.add_argument("--username", required=True)
    create_cmd.add_argument("--password", required=True)
    create_cmd.add_argument("--email", default=None)

    sub.add_parser("list", help="List admin users.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        create_or_update_admin(args.username, args.password, args.email)
        return
    if args.command == "list":
        list_admins()
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
