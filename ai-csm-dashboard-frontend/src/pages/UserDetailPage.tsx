// src/pages/UserDetailPage.tsx
import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
// import { Button } from '@/components/ui/button'; // Removed as it's not used on this page currently
import { ArrowLeft } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Interfaces
interface User {
  id: number;
  email: string;
  company_id: number;
  created_at: string;
  company_name?: string;
}

interface Company {
  id: number;
  name: string;
}

interface UserEvent {
  id: number;
  event_name: string;
  properties: Record<string, any> | null;
  timestamp: string;
}

interface UserIntervention {
  id: number;
  intervention_type: string;
  channel: string;
  content_sent: string;
  status: string;
  sent_at: string;
}

interface UserHealthScore {
  id: number;
  score: number;
  status: string;
  calculated_at: string;
}

const UserDetailPage = () => {
  const { userId } = useParams<{ userId: string }>();
  const [user, setUser] = useState<User | null>(null);
  const [company, setCompany] = useState<Company | null>(null);
  const [events, setEvents] = useState<UserEvent[]>([]);
  const [interventions, setInterventions] = useState<UserIntervention[]>([]);
  const [healthScores, setHealthScores] = useState<UserHealthScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null); // For critical page load errors

  useEffect(() => {
    if (!userId) {
      setError("User ID is missing.");
      setLoading(false);
      return;
    }

    const fetchUserData = async () => {
      setLoading(true);
      setError(null); // Reset error before new fetch attempt
      try {
        // --- Fetch user details (Primary fetch) ---
        const userRes = await fetch(`${API_BASE_URL}/users/${userId}`);
        if (!userRes.ok) {
          if (userRes.status === 404) {
            throw new Error(`User with ID ${userId} not found.`);
          }
          throw new Error(`Failed to fetch user details. Status: ${userRes.status}`);
        }
        const userData: User = await userRes.json();
        setUser(userData);

        // --- Fetch associated data (Secondary fetches, handle 404 gracefully) ---
        const fetchDataForUser = async (endpoint: string, setter: React.Dispatch<React.SetStateAction<any[]>>, entityName: string) => {
          try {
            const res = await fetch(`${API_BASE_URL}/users/${userId}/${endpoint}`);
            if (res.ok) {
              const data = await res.json();
              setter(data);
            } else if (res.status === 404) {
              setter([]); // Treat 404 as empty list for this sub-entity
              console.log(`No ${entityName} found for user ${userId} (404), setting to empty.`);
            } else {
              console.error(`Failed to fetch ${entityName} for user ${userId}. Status: ${res.status}`);
              setter([]); // Default to empty on other errors for this sub-entity
            }
          } catch (fetchError) {
            console.error(`Error fetching ${entityName} for user ${userId}:`, fetchError);
            setter([]); // Default to empty on network or parsing errors
          }
        };
        
        // Fetch company details (slightly different pattern as it's not a list under user)
        if (userData.company_id) {
          try {
            const companyRes = await fetch(`${API_BASE_URL}/companies/${userData.company_id}`);
            if (companyRes.ok) {
              const companyData: Company = await companyRes.json();
              setCompany(companyData);
              // Update user object with company name if fetched (or rely on dashboard passing it)
              setUser(prevUser => prevUser ? { ...prevUser, company_name: companyData.name } : null);
            } else {
              console.warn(`Could not fetch company details for company_id ${userData.company_id} (status: ${companyRes.status})`);
            }
          } catch (companyError) {
             console.error(`Error fetching company details for company_id ${userData.company_id}:`, companyError);
          }
        }

        // Use Promise.all to fetch events, interventions, and health scores concurrently
        await Promise.all([
          fetchDataForUser('events/?limit=20', setEvents, 'events'),
          fetchDataForUser('interventions/?limit=10', setInterventions, 'interventions'),
          fetchDataForUser('health_scores/?limit=5', setHealthScores, 'health scores')
        ]);

      } catch (err) { // This catch block will primarily handle errors from the main user fetch
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('An unknown error occurred while loading user data.');
        }
        console.error("Critical error in UserDetailPage fetchUserData:", err);
        setUser(null); // Ensure user is null if the main fetch fails
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [userId]);

  const getStatusColorClasses = (status?: string): string => {
    switch (status) {
      case 'At Risk': return 'bg-red-300 hover:bg-red-400 text-red-800';
      case 'Power User': return 'bg-green-200 hover:bg-green-300 text-green-800';
      case 'Healthy': return 'bg-blue-300 hover:bg-blue-400 text-blue-800';
      case 'Neutral': return 'bg-gray-300 hover:bg-gray-400 text-gray-800';
      default: return 'bg-gray-100 hover:bg-gray-200 text-gray-800';
    }
  };

  if (loading) return <div className="flex justify-center items-center h-screen"><p>Loading user details...</p></div>;
  
  // If there's a critical error (e.g., user not found), display the error message
  if (error) return <div className="text-red-500 p-4 bg-red-100 border border-red-300 rounded text-center">Error: {error}</div>;
  
  // If loading is done and user is still null (and no error was set, though unlikely with current logic), means user not found.
  // The error state should ideally cover this if the main user fetch fails.
  if (!user) return <div className="text-center p-8 text-muted-foreground">User data could not be loaded or user not found.</div>;

  return (
    <div className="space-y-6">
      <Link to="/" className="inline-flex items-center text-sm font-medium text-primary hover:underline mb-4">
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Dashboard
      </Link>

      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">{user.email}</CardTitle>
          <CardDescription>
            User ID: {user.id} | Company: {user.company_name || company?.name || `ID ${user.company_id}`} | Joined: {new Date(user.created_at).toLocaleDateString()}
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader><CardTitle>Recent Health Scores</CardTitle></CardHeader>
        <CardContent>
          {healthScores.length > 0 ? (
            <Table>
              <TableHeader><TableRow><TableHead>Score</TableHead><TableHead>Status</TableHead><TableHead>Calculated At</TableHead></TableRow></TableHeader>
              <TableBody>
                {healthScores.map(hs => (
                  <TableRow key={hs.id}>
                    <TableCell>{hs.score}</TableCell>
                    <TableCell><Badge className={getStatusColorClasses(hs.status)}>{hs.status}</Badge></TableCell>
                    <TableCell>{new Date(hs.calculated_at).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : <p className="text-sm text-muted-foreground">No health scores recorded.</p>}
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader><CardTitle>Recent Events ({events.length})</CardTitle></CardHeader>
        <CardContent>
          {events.length > 0 ? (
            <Table>
              <TableHeader><TableRow><TableHead>Event Name</TableHead><TableHead>Properties</TableHead><TableHead>Timestamp</TableHead></TableRow></TableHeader>
              <TableBody>
                {events.map(event => (
                  <TableRow key={event.id}>
                    <TableCell>{event.event_name}</TableCell>
                    <TableCell className="text-xs max-w-xs truncate">
                      {event.properties ? JSON.stringify(event.properties) : 'N/A'}
                    </TableCell>
                    <TableCell>{new Date(event.timestamp).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : <p className="text-sm text-muted-foreground">No recent events.</p>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Recent Interventions ({interventions.length})</CardTitle></CardHeader>
        <CardContent>
          {interventions.length > 0 ? (
            <Table>
              <TableHeader><TableRow><TableHead>Type</TableHead><TableHead>Channel</TableHead><TableHead>Status</TableHead><TableHead className="max-w-md">Content (snippet)</TableHead><TableHead>Sent At</TableHead></TableRow></TableHeader>
              <TableBody>
                {interventions.map(intervention => (
                  <TableRow key={intervention.id}>
                    <TableCell>{intervention.intervention_type}</TableCell>
                    <TableCell>{intervention.channel}</TableCell>
                    <TableCell><Badge variant={intervention.status === 'failed' ? 'destructive' : 'secondary'}>{intervention.status}</Badge></TableCell>
                    <TableCell className="text-xs max-w-md truncate" title={intervention.content_sent}>
                      {intervention.content_sent.substring(0, 100)}{intervention.content_sent.length > 100 ? '...' : ''}
                    </TableCell>
                    <TableCell>{new Date(intervention.sent_at).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : <p className="text-sm text-muted-foreground">No recent interventions.</p>}
        </CardContent>
      </Card>
    </div>
  );
};

export default UserDetailPage;