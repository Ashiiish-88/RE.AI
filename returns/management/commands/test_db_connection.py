from django.core.management.base import BaseCommand
from django.db import connection
from django.conf import settings
import time
import sys


class Command(BaseCommand):
    help = 'Test database connection with progress feedback'

    def handle(self, *args, **options):
        self.stdout.write('🔄 Starting database connection test...')
        
        # Show current database configuration
        db_config = settings.DATABASES['default']
        if 'postgresql' in db_config.get('ENGINE', ''):
            self.stdout.write('📊 Database: PostgreSQL (Supabase)')
        else:
            self.stdout.write('📊 Database: SQLite (fallback)')
            
        self.stdout.write('⏳ Attempting to connect...')
        
        try:
            # Add timeout and progress indication
            self.stdout.write('🔌 Testing connection...', ending='')
            self.stdout.flush()
            
            with connection.cursor() as cursor:
                db_engine = settings.DATABASES['default']['ENGINE']
                
                if 'postgresql' in db_engine:
                    self.stdout.write(' ✅')
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    self.stdout.write(
                        self.style.SUCCESS('🎉 Successfully connected to PostgreSQL (Supabase)!')
                    )
                    self.stdout.write(f'📋 Version: {version}')
                    
                    cursor.execute("SELECT current_database(), current_user")
                    db_info = cursor.fetchone()
                    self.stdout.write(f'🗄️  Database: {db_info[0]}')
                    self.stdout.write(f'👤 User: {db_info[1]}')
                    
                elif 'sqlite' in db_engine:
                    self.stdout.write(' ✅')
                    cursor.execute("SELECT sqlite_version()")
                    version = cursor.fetchone()[0]
                    self.stdout.write(
                        self.style.WARNING('⚠️  Using SQLite (Supabase not configured)')
                    )
                    self.stdout.write(f'📋 SQLite version: {version}')
                    
        except Exception as e:
            self.stdout.write(' ❌')
            self.stdout.write(
                self.style.ERROR(f'💥 Connection failed: {str(e)}')
            )
            
            # Provide troubleshooting hints
            if 'could not translate host name' in str(e):
                self.stdout.write('💡 Hint: DNS resolution failed. Check your internet connection.')
            elif 'timeout' in str(e).lower():
                self.stdout.write('💡 Hint: Connection timeout. Check firewall settings.')
            elif 'authentication failed' in str(e).lower():
                self.stdout.write('💡 Hint: Check your username/password in .env file.')
                
            return
            
        self.stdout.write(
            self.style.SUCCESS('✨ Database connection test completed!')
        )
